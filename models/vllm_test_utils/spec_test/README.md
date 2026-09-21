# Speculative-decoding test model

`DummySpecDecodeModel` supplies host arithmetic for the plugin's speculative
verification and model-owned drafting interfaces. A real vLLM server can use
`DummySpecDecodeModel` to test scheduling, acceptance, output bookkeeping, and
host overhead without model weights or device kernels. The TT worker still
opens a mesh device, so server runs require the normal device environment.

The shared interface is documented in
[`vllm-tt-plugin` issue #110](https://github.com/tenstorrent/vllm-tt-plugin/issues/110)
and `docs/SPEC_DECODE_CONTRACT.md` in the plugin repository. Use a plugin revision
that implements model-owned speculative execution and registers
`TTDummySpecDecodeModel` when `register_test_models` is enabled.

## Target, proposal policy, and capacity

Set these environment variables before starting the server. The plugin reads
some declarations from the model class before constructing an instance.

- `TT_SPEC_TARGET=depth` (default): verification copies candidate successors up
  to the requested acceptance depth. This target checks acceptance accounting;
  ordinary and speculative generation need not produce equal sequences.
- `TT_SPEC_TARGET=fixed`: prefill, ordinary decode, and verification use the
  successor rule `(token * 31 + position * 7 + 11) % vocab`. Each verification
  column reads its candidate input token, which can be a preceding draft. The
  rule does not copy the candidate successor being scored. Use this target in
  both arms of an ordinary/speculative output-sequence comparison.
- `TT_SPEC_ACCEPT_DEPTH=-1` (default): accept all valid model-owned drafts.
  `0` requests zero accepted drafts; a nonnegative `n` requests at most the
  first `n` valid drafts per row. For `depth`, verification enforces the depth.
  For `fixed`, the model-owned drafter changes proposals beyond the depth;
  the fixed target rule itself is unchanged. The depth does not control
  acceptance of an unrelated n-gram proposer under the fixed target.
- `TT_SPEC_DRAFT_POLICY=always` (default): the model-owned drafter proposes K
  tokens after each commit. A steady-state commit contains the accepted valid
  prefix plus a correction or bonus. The first decode has no pending drafts,
  and request/context limits can shorten later commits.
- `TT_SPEC_DRAFT_POLICY=solo`: the model-owned drafter proposes K tokens when
  exactly one request is live and reports `DraftOutput.num_valid=0` otherwise.
  `solo` also enables `supports_narrow_decode` and removes the `hidden_feed`
  requirement, allowing eligible draftless steps to use ordinary decode.
  This policy tests transitions; it does not predict a real model's performance.
- `TT_SPEC_MAX_TOKENS_ALL_USERS=131072` (default): logical KV token capacity
  reported by `get_max_tokens_all_users`. A smaller value can make scheduler
  preemption reachable. The model allocates no device KV cache. The plugin
  derives `num_gpu_blocks_override` from this capacity, so use this environment
  variable for capacity tests rather than `--num-gpu-blocks-override`.

`solo` counts live requests from nonnegative committed positions, not padded
batch rows. The drafter always returns a `[B, K]` tensor; `num_valid` specifies
which proposal prefixes are usable. `always` retains the hidden-handoff check:
`propose_draft_tokens` must receive the most recent verification's opaque
Python object. Ordinary decode retains token, position, and page-table inputs
for subsequent submissions. Verification arithmetic and fixed-target proposal
arithmetic use the supplied inputs rather than resident ordinary decode state.

## Synchronous model-owned launch

This command selects a controlled synchronous run. `--no-async-scheduling` is
an explicit choice for this example, not a limitation of `DummySpecDecodeModel`.

```bash
TT_SPEC_TARGET=depth TT_SPEC_DRAFT_POLICY=always TT_SPEC_ACCEPT_DEPTH=-1 \
vllm serve models/vllm_test_utils/spec_test \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --additional-config '{"tt": {"register_test_models": true}}' \
    --speculative-config '{"method": "custom_class",
                           "model": "vllm_tt_plugin.model_owned_drafter",
                           "num_speculative_tokens": 5}' \
    --max-num-seqs 8 \
    --no-async-scheduling
```

Upstream vLLM uses `custom_class` for a user-provided proposer class specified
by a dotted Python path. On the TT model-owned path,
`vllm_tt_plugin.model_owned_drafter` is a required compatibility marker.
The TT runner does not import that marker or load a second model from it;
the TT runner calls `propose_draft_tokens` on the loaded `DummySpecDecodeModel`.
The actual serving model is still `models/vllm_test_utils/spec_test`.

Send greedy requests (`temperature: 0`) without logprobs, structured output,
token filters, or penalties. The paired plugin's implemented speculative path
uses `argmax_ids` and rejects unsupported sampling semantics. A normal text
prompt is sufficient for the model-owned drafter.

## Asynchronous scheduling

`DummySpecDecodeModel` declares both capabilities below. They have separate
requirements, and the plugin evaluates them in this order:

1. `TTPlatform` checks `supports_async_decode`. If asynchronous scheduling is
   requested and `supports_async_decode` is absent or false, `TTPlatform`
   disables asynchronous scheduling. This capability covers ordinary decode:
   split submission/readback and resident input updates under
   `decode_input_update_contract=1`.
2. If speculation is configured and asynchronous scheduling remains enabled,
   `TTPlatform` requires `supports_async_spec_decode`. If this declaration is
   absent or false, `TTPlatform` rejects the combination. The speculative
   capability covers deferred readback of a `[B, 1+K]` verification result and
   retention of the hidden handle until the dependent proposal.

`supports_async_spec_decode` does not imply `supports_async_decode` or enable
asynchronous scheduling by itself. `supports_async_decode` does not establish
speculative output or hidden-state lifetime safety. Other plugin admission
checks still apply after these capability checks.

For a model-owned asynchronous run, use the preceding command with
`TT_SPEC_TARGET=fixed`, `TT_SPEC_DRAFT_POLICY=solo`, and `--async-scheduling`
in place of `--no-async-scheduling`. Replace `--additional-config` with:

```bash
--additional-config '{"tt": {"register_test_models": true, "sample_on_device_mode": "decode_only"}}'
```

`sample_on_device_mode="decode_only"` enables the decode token-feedback path
required for eligible ordinary overlap. Without device sampling on decode,
`can_use_steady_decode_fast_path` rejects overlap even when asynchronous
scheduling is enabled. The dummy returns host token IDs through this interface;
no device sampling kernel runs. Use a plugin revision with the guarded TT
`custom_class` asynchronous configuration support. A speculative verify is
serialized: the plugin drains outstanding ordinary work before verification,
and acceptance/commit must finish before the dependent proposal. Eligible
ordinary draftless decodes can overlap host scheduling and output processing.
Enabling asynchronous scheduling does not run draft and verification in
parallel, and `always` does not provide the same ordinary overlap opportunity.

`DummySpecDecodeModel` executes immediately on the host and
`read_decode_output` returns no device completion events. These runs exercise
scheduling interfaces, not TT kernel overlap, device buffer lifetime, real KV
state selection, or real-model speedup. Plugin host tests with controlled
completion events check delayed-readback ordering separately.

## Host n-gram launch

This command tests target verification with vLLM's n-gram proposals. It does
not exercise `DummySpecDecodeModel.propose_draft_tokens` or its hidden check.

```bash
TT_SPEC_TARGET=depth TT_SPEC_DRAFT_POLICY=always TT_SPEC_ACCEPT_DEPTH=-1 \
vllm serve models/vllm_test_utils/spec_test \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --additional-config '{"tt": {"register_test_models": true}}' \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 5,
                           "prompt_lookup_min": 2, "prompt_lookup_max": 4}' \
    --max-num-seqs 8 \
    --no-async-scheduling
```

Under these explicit settings, prefill chooses token 0 and draftless wide
verification increments the committed token. An ascending token-ID prompt
supplies repeated n-grams for the generated run to find:

```bash
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{
  \"model\": \"models/vllm_test_utils/spec_test\",
  \"prompt\": [$(seq -s, 0 399)],
  \"max_tokens\": 200,
  \"temperature\": 0
}"
```

Choose a prompt run long enough to cover the requested output and complete
lookahead blocks. Inspect actual proposal counts; ordinary text does not
reliably contain the ascending sequences this target generates. When the
n-gram proposer returns no drafts under `always`, the plugin still sends a
wide verification block with `num_valid_drafts=0`. The acceptance walk commits
one token, so a successful response alone does not prove drafts were tested.

## Measurement and correctness evidence

For a steady-state host-overhead measurement, use model-owned drafting with
`TT_SPEC_TARGET=depth` and `TT_SPEC_DRAFT_POLICY=always`. Record K, acceptance
depth, actual valid drafts, batch size, request/context limits, and the timing
boundary. Exclude or report bootstrap and clipped terminal steps separately.
The measurement includes dummy tensor construction, plugin acceptance,
scheduling, and proposal work. It is not vLLM-only cost or real-device latency.
N-gram measurements also include proposer work and may include first-call
compilation unless warmup is excluded.

For correctness, compare complete token-ID sequences with and without
speculation using `TT_SPEC_TARGET=fixed` in both arms. The fixed target's
prefill rule also supports replay after preemption. Verify lifecycle events
explicitly: capacity pressure alone does not prove preemption, and closing a
client connection alone does not prove cancellation completed.

Require positive proposal counts and observed speculative verification to prove
the intended path ran. When acceptance is expected, require at least one
non-bootstrap commit longer than one token. At acceptance depth zero, require
rejections and one-token commits instead. HTTP success cannot establish any of
these properties.

## Tests and CI integration

`models/vllm_test_utils/tests/host/test_spec_test_model.py` covers verification
layout, per-row acceptance, draft construction, hidden identity, fixed-target
arithmetic, solo policy, resident ordinary decode, and prefill replay. The full
suite requires `vllm_tt_plugin` on `PYTHONPATH` because contract calls import its
validators and value types. Isolated arithmetic helpers need no TT device.

The paired plugin's `tests/tt/spec/` suite exercises this model through a real
server. Running host tests or a successful server request does not substitute
for that suite's proposal, acceptance, lifecycle, and exact-output assertions.

The PR does not register a model entry in
`tests/pipeline_reorg/vllm_model_tests.yaml` or a dedicated host-test CI job.
CI integration must select a compatible plugin revision, make both repositories
importable, collect the host tests explicitly, and run server checks that prove
the requested paths. `.github/workflows/vllm-model-tests.yaml` supports a plugin
ref, but its default `main` must not be assumed to include the required stack.
A generic text benchmark is insufficient for the n-gram target scenario above.
