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
when the TT config asks for the test models, so `--additional-config` below is
required.

There are two launches, because this model serves both halves of the contract
and the drafting method decides which one runs.

**The model's own drafter**, which is what exercises `propose_draft_tokens` and
the hidden-state handoff. Prefer this one: it speculates on every step after
the first whatever the prompt says.

```bash
vllm serve models/vllm_test_utils/spec_test \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --additional-config '{"tt": {"register_test_models": true}}' \
    --speculative-config '{"method": "custom_class",
                           "model": "vllm_tt_plugin.model_owned_drafter",
                           "num_speculative_tokens": 5}' \
    --max-num-seqs 8 \
    --no-async-scheduling
```

`custom_class` is vLLM's name for a proposer it does not own, which is what a
model-owned drafter is, and the `model` key must be exactly
`vllm_tt_plugin.model_owned_drafter`: vLLM requires a dotted proposer path
there, nothing imports it, and the plugin refuses any other value rather than
letting a path that goes nowhere look meaningful.

**The host n-gram drafter**, which asks this model for no drafting at all and
exercises the verify alone.

```bash
vllm serve models/vllm_test_utils/spec_test \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --additional-config '{"tt": {"register_test_models": true}}' \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 5,
                           "prompt_lookup_min": 2, "prompt_lookup_max": 4}' \
    --max-num-seqs 8 \
    --no-async-scheduling
```

`--no-async-scheduling` is not optional for this model, for a reason that is
not about speculation. The plugin disables asynchronous scheduling for any
model that does not declare `model_capabilities['supports_async_decode']`, and
this model does not: it has no split submission and readback and recomputes
from the host tokens of every step. Passing the flag makes the launch say what
it is going to do.

The speculative half of the asynchronous path is declared: this model sets
`model_capabilities['supports_async_spec_decode']`, which the plugin requires
before it admits speculation together with asynchronous scheduling. That
declaration is about two things the deferred path does to a model: it hands
`read_decode_output` a `[B, 1+K]` verify whose committed length the host
decides after the forward, and it holds the verify's hidden handle across the
readback until the next step's propose call. This model satisfies both
trivially, because it returns host tensors and a host hidden object. The
property worth asserting here is the one the host tests cover: it keeps no
state between steps. The declaration does not make this model run
asynchronously on a server, because the ordinary async-decode declaration
above is the one the plugin reads first.

Requests must be greedy. The `ngram` method with `argmax_ids` acceptance
compares token ids and never sees logits, so the plugin refuses a request
carrying a temperature, logprobs, structured output, a token filter or a
penalty, rather than answering it greedily without saying so.

This needs a `vllm-tt-plugin` revision whose speculative execution path has
landed. On a revision without it the launch is refused at configuration time,
which is the intended behaviour there and not a fault of this model.

## With the n-gram drafter, the prompt decides whether speculation runs at all

This section is about the n-gram launch only. The model's own drafter proposes
from each row's last committed token, so any prompt speculates.

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
server commits one token per step. That still exercises configuration
admission, the KV cache, the engine's `take_draft_token_ids` handshake and the
accept walk, which runs with `num_valid_drafts` of 0 on every row: this model's
`spec_plan` returns `supports_narrow_decode=False`, so the candidate block
keeps its full `[B, 1+K]` width even with nothing drafted, and the narrow call
is never made. What it does not exercise is a single accepted draft. Driven on
the host at `num_speculative_tokens=3` over ten decode steps, the run prompt
commits 37 tokens and a text prompt commits 10.

## The two modes

| `TT_SPEC_ACCEPT_DEPTH` | What happens | What it is for |
| --- | --- | --- |
| unset, or -1 | every draft is accepted | exercises the loop hardest: the longest committed prefix, the widest commit, the most placeholder accounting |
| `0` | no draft is accepted | each step commits one token, so the step's wall clock is the host cost of one whole iteration |
| `n` | the first `n` drafts of each row are accepted | a predictable accepted length, for checking the output bookkeeping against a known answer |

The depth is per row and never reduced across the batch, so a row that carries
fewer drafts than another does not shorten it.

## The two draft policies

`TT_SPEC_DRAFT_POLICY` decides how much this model's drafter offers, and it is
a separate axis from the accept depth above: the depth says what the verify
accepts, the policy says what the drafter proposes.

| `TT_SPEC_DRAFT_POLICY` | What the drafter offers | What it is for |
| --- | --- | --- |
| unset, or `always` | the full draft length on every step | a fixed committed width, which is what a per-step cost has to divide cleanly |
| `solo` | the full length while one request is live, nothing while more are | the adaptive shape a real deployment has, where speculation pays for a lone request and loses to batching for a full one |

The `solo` policy says "nothing" with `DraftOutput.num_valid` at 0 for the
row, never by returning fewer ids or by filling them with a token it hopes the
runner reads as empty: the id tensor is `[B, K]` whatever happens, because a
device graph has one shape. It counts live requests from the committed
positions and not from the number of rows, because the rows are padded to the
wire batch size and a padding row's positions are negative for exactly this.

Two declarations move with it, because the plugin reads them off the class at
configuration time. `spec_plan` reports `supports_narrow_decode=True`, which is
what lets the plugin send a step with nothing to verify as this model's own
ordinary decode, and those steps are the ones that can overlap. And the
drafter stops requiring a fed hidden state: such a step returns no
`VerifyOutput` and so no hidden handle, and the plugin keeps a drafter that
needs one on the verify path, which would defeat the policy. Under this policy
the drafter continues from the committed block alone.

What the policy is for, concretely: a server with speculation configured used
to run every decode step as a verify, and a verify is never overlapped, so
configuring speculation cost the server its asynchronous batched decoding.
With this policy the batched steps are ordinary overlapping decodes and the
solo steps speculate.

## What the measurement mode measures

The cost of the whole loop with no device work in it: the scheduler, the
candidate-block build, this model's arithmetic, the plugin's acceptance walk,
the commit, and the proposal.

Measure with the model's own drafter rather than with n-gram. Every step then
commits the same width, so a per-step cost divides cleanly, and there is no
`numba` compilation in the run: the n-gram proposer's first call pays about a
second to compile `batch_propose_numba`, which lands on the first speculative
step and is easy to mistake for a per-step cost.

It is not vLLM's cost alone. This model builds a `[B, 1+K]` answer on every
step, which measures about 17 microseconds at every shape from `[1, 6]` to
`[32, 17]`, against a step whose other costs are milliseconds.

## Continuous integration

Nothing automated runs this model or its host tests yet, and both items are
blocked on the same thing: a speculative launch needs a `vllm-tt-plugin`
revision whose execution path has landed, and
`.github/workflows/vllm-model-tests.yaml` defaults its plugin ref to `main`,
where such a launch is refused at configuration time.

Two things to add once a revision with it is selectable.

**The host tests.** `models/vllm_test_utils/tests/host/test_spec_test_model.py`
needs `vllm_tt_plugin` importable, because the drafter validates its inputs
against the contract module's own validator and returns the contract's
`DraftOutput`. No tt-metal job collects `models/**/tests/host` today, and the
one workflow that does check out the plugin and put both repositories on
`PYTHONPATH` is the device workflow above, so this needs either a host step in
that workflow or a plugin-equipped host job of its own. Until then these tests
run by hand, with the plugin on `PYTHONPATH`.

**The server entries**, in `tests/pipeline_reorg/vllm_model_tests.yaml`,
alongside the `no_op_test` one. The model-owned drafter first, because it is
the one that exercises `propose_draft_tokens` and the hidden handoff:

```yaml
- name: "Speculative decode test model, model-owned drafter"
  model: models/vllm_test_utils/spec_test
  server-timeout: 2
  benchmark-timeout: 2
  mesh-device: "(1, 1)"
  arch: arch-wormhole_b0
  tt-config: '{"register_test_models": true, "input_queue_batching_delay": 0}'
  additional-server-args: "--tokenizer meta-llama/Llama-3.1-8B-Instruct --no-async-scheduling --speculative-config {\"method\":\"custom_class\",\"model\":\"vllm_tt_plugin.model_owned_drafter\",\"num_speculative_tokens\":5}"
  additional-benchmark-args: "--tokenizer meta-llama/Llama-3.1-8B-Instruct --temperature 0.0"
  multimodal: false
```

And the n-gram one, which exercises the verify against a host drafter:

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

A request completing is not enough to prove speculation ran: either run
completes whether or not a draft was ever proposed or accepted. The assertion
has to be that drafts were proposed and that at least one step after a
request's first committed more than one token. A request's first decode step
always commits exactly one, because the drafter runs after a commit and nothing
is in flight yet, so an assertion that every step commits a wide prefix fails
on a correct run.

The n-gram entry needs its prompt chosen as well: the benchmark's own prompts
cannot produce a draft against this model, whether random or sampled from a
dataset, so it has to send the ascending run above. The model-owned entry has
no such requirement.
