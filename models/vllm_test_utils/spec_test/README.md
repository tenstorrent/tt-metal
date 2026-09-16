# Speculative-decoding test model

`DummySpecDecodeModel` implements `vllm-tt-plugin`'s speculative-decoding
contract and does no device work. It exists to run the plugin's speculative
path against a real vLLM server, and to measure what one verify-then-propose
iteration costs on the host.

The contract it implements is
<https://github.com/tenstorrent/vllm-tt-plugin/issues/110>, and the model side
of it is `docs/SPEC_DECODE_CONTRACT.md` in that repository.

## Running it

The architecture name is `TTDummySpecDecodeModel`. The plugin registers it only
when the TT config asks for the test models, so both of these are required:

```bash
vllm serve models/vllm_test_utils/spec_test \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --additional-config '{"tt": {"register_test_models": true}}' \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 5,
                           "prompt_lookup_min": 2, "prompt_lookup_max": 4}' \
    --max-num-seqs 8 \
    --no-async-scheduling
```

`--no-async-scheduling` is not optional. The plugin's accept walk runs in the
synchronous decode tail, and it refuses a launch that combines speculation with
asynchronous scheduling rather than taking a path that would skip acceptance.

Requests must be greedy. The `ngram` method with `argmax_ids` acceptance
compares token ids and never sees logits, so the plugin refuses a request
carrying a temperature, logprobs, structured output, a token filter or a
penalty, rather than answering it greedily without saying so.

This needs a `vllm-tt-plugin` revision whose speculative execution path has
landed. On a revision without it the launch is refused at configuration time,
which is the intended behaviour there and not a fault of this model.

## The prompt decides whether speculation runs at all

`DummyNoOpModel.prefill_forward` returns zero logits, so the first sampled token
is id 0 whatever the prompt says, and a draftless step of this model commits
`last_committed + 1`. Its output is therefore the ascending run 0, 1, 2, 3, and
so on, which repeats no n-gram. The n-gram proposer finds nothing in that output
and drafts only what the prompt puts in reach, so the prompt has to be the same
run, sent as explicit token ids:

```bash
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{
  \"model\": \"models/vllm_test_utils/spec_test\",
  \"prompt\": [$(seq -s, 0 399)],
  \"max_tokens\": 200,
  \"temperature\": 0
}"
```

The generated stream re-enters that run at position 0 on its own, so from the
second decode step onward every step is offered `num_speculative_tokens` drafts
and commits all of them plus the bonus. Speculation stops once the output passes
the end of the run, so the run has to be at least `max_tokens + 2` long.

A natural-language prompt drafts nothing. The pair the output ends on never
appears earlier in the sequence, so every step is a draftless step and the
server commits one token per step. That exercises configuration admission, the
KV cache, the engine's `take_draft_token_ids` handshake and the narrow path, and
no part of acceptance. Driven on the host at `num_speculative_tokens=3` over ten
decode steps, the run prompt commits 37 tokens and a text prompt commits 10.

## The two modes

| `TT_SPEC_ACCEPT_DEPTH` | What happens | What it is for |
| --- | --- | --- |
| unset, or -1 | every draft is accepted | exercises the loop hardest: the longest committed prefix, the widest commit, the most placeholder accounting |
| `0` | no draft is accepted | each step commits one token, so the step's wall clock is the host cost of one whole iteration |
| `n` | the first `n` drafts of each row are accepted | a predictable accepted length, for checking the output bookkeeping against a known answer |

The depth is per row and never reduced across the batch, so a row that carries
fewer drafts than another does not shorten it.

## What the measurement mode measures

The cost of the whole loop with no device work in it: the scheduler, the
candidate-block build, this model's arithmetic, the plugin's acceptance walk,
the commit, and the n-gram proposal.

It is not vLLM's cost alone. This model builds a `[B, 1+K]` answer on every
step, which measures about 17 microseconds at every shape from `[1, 6]` to
`[32, 17]`, against a step whose other costs are milliseconds.

## Continuous integration

Not registered in `tests/pipeline_reorg/vllm_model_tests.yaml` yet, because a
speculative launch needs a plugin revision that can execute one and the
workflow defaults its plugin ref to `main`. The entry to add once that lands,
alongside the `no_op_test` one:

```yaml
- name: "Speculative decode test model (vLLM overhead)"
  model: models/vllm_test_utils/spec_test
  server-timeout: 2
  benchmark-timeout: 2
  mesh-device: "(1, 1)"
  arch: arch-wormhole_b0
  tt-config: '{"register_test_models": true, "input_queue_batching_delay": 0}'
  additional-server-args: "--tokenizer meta-llama/Llama-3.1-8B-Instruct --no-async-scheduling --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_min\":2,\"prompt_lookup_max\":4}"
  additional-benchmark-args: "--tokenizer meta-llama/Llama-3.1-8B-Instruct --temperature 0.0"
  multimodal: false
```

A request completing is not enough to prove speculation ran: an n-gram run
completes whether or not it ever proposed a draft. The assertion has to be that
drafts were proposed and a prefix longer than one token committed. The
benchmark's own prompts cannot produce a draft against this model, whether they
are random or sampled from a dataset, so an entry that measures the speculative
loop rather than the draftless path has to send the ascending run above.
