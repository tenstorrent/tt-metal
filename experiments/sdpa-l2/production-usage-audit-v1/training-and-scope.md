# Training and residual repository scope

Static audit of the current working tree, 2026-09-16. Decode operator variants
are excluded; invoking a non-decode operator from a decode-like model phase
does not automatically exclude that invocation. No device tests were run.

## TT-Train is an independent attention implementation

The Python census finds 14 TTML attention call/reference sites in nine files.
These are **not** calls into TTNN's inference SDPA implementation. The full
site list is in `python-census.json`, family `ttml`:

| File under `tt-train/` | Lines | Use |
|---|---|---|
| `sources/examples/grpo/utils/llama_overrides.py` | 27, 71 | Composite attention |
| `sources/examples/grpo_remote_rollout/utils/llama_overrides.py` | 32, 76 | Composite attention |
| `sources/examples/nano_gpt/nanogpt_primitives_example.py` | 240 | Fused training attention |
| `sources/examples/qwen3/model_qwen3_distributed.py` | 288, 290 | Fused/composite function selection |
| `sources/ttml/ttml/models/deepseek/mla.py` | 151 | Fused training attention, unequal QK/V widths |
| `sources/ttml/ttml/models/llama/gqattn.py` | 120, 170 | GQA including cached attention |
| `sources/ttml/ttml/models/nanogpt/multi_head_attention.py` | 63 | Fused training attention |
| `sources/ttml/ttml/models/qwen3/attention.py` | 124, 126 | Fused/composite function selection |
| `tests/python/test_deepseek_mla_sdpa.py` | 87 | Composite reference |

The C++ module call sites are:

- `sources/ttml/modules/single_head_attention.cpp:32`
- `sources/ttml/modules/multi_head_attention.cpp:32`
- `sources/ttml/modules/grouped_query_attention.cpp:50,104`
- `sources/ttml/modules/distributed/multi_head_attention.cpp:47`
- `sources/ttml/modules/distributed/grouped_query_attention.cpp:117,120`

The wrappers are bound at `sources/ttml/nanobind/nb_ops.cpp:359,377`.
`sources/ttml/ops/scaled_dot_product_attention.cpp:258` invokes
`ttml::metal::sdpa_fw`, retaining FP32 log-sum-exp intermediates for its
backward kernel. Distributed training calls the local wrapper at
`sources/ttml/ops/distributed/ring_attention_sdpa.cpp:64` or its own
`ttml::metal::ring_sdpa_fw` at line 137. The ring factory delegates to the
training forward factory (`metal/ops/ring_sdpa_fw/device/ring_sdpa_fw_program_factory.cpp:76`).

Fused training input validation requires tiled **BF16 Q/K/V**, and BF16 masks
where supplied (`metal/ops/sdpa_fw/device/sdpa_fw_device_operation.cpp:60–62,121`).
The forward factory fixes **FP32 destination**, **math approximation off**,
and chooses **HiFi3 on Wormhole / HiFi4 otherwise**. See
`metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp:369,537–540,613–616`.
The source explicitly cites the Wormhole HiFi4+FP32-destination matmul issue
as the reason for its HiFi3 choice. This is source evidence of a compatibility
concern, not a new hardware reproduction by this audit.

Composite attention instead uses separate matmul/softmax operations
(`ops/scaled_dot_product_attention.cpp:140` onwards). Its matmul wrapper uses
HiFi4, FP32 destination, math approximation off, packer L1 accumulation on
(`ttnn_fixed/matmuls.cpp:29`, `core/compute_kernel_config.cpp:27–33`). Its input
dtype is inherited from the model/autograd tensor, not a newly imposed
E/G preparation contract.

C++ test/operator call sites include:

- `tests/utils/memory_utils_test.cpp:182`
- `tests/ops/distributed/ring_sdpa_test.cpp:327`
- `tests/ops/sdpa_fw_op_test.cpp:535,840,859`
- `tests/ops/sdpa_bw_op_test.cpp:596`

**Migration implication:** no caller updates are required merely to refactor
TTNN inference SDPA. These independent training forward/backward/composite
implementations should not be deleted under the label "remove non-streaming
SDPA." Unifying them would be a separate project with autograd, dropout and
intermediate-output contracts, not the proposed inference migration.

## Documentation and tooling

- `docs/source/ttnn/ttnn/api.rst:508–523` indexes public attention functions,
  including decode entries that remain out of scope. Add preset documentation
  without removing the existing non-decode functions prematurely.
- `tech_reports/ViT-TTNN/vit_bh.md:1130–1165` documents a real ViT call using
  HiFi4, FP32 destination, math/exp approximation off, packer L1 on and L1
  output. This agrees with the model audit; it is not an independent caller.
- `scripts/detect_undocumented_ttnn_ops.py:69` lists experimental ring-joint
  SDPA as documentation-exempt; it is metadata rather than an execution site.
- `scripts/detect_override_rebuild.py` mentions sparse SDPA as an example of
  cache-hit runtime-argument patching, not an SDPA caller.
- The LLM/FlashDecode tech reports contain decode explanations and model
  links, not additional non-decode precision recipes.

## Census scope and limits

`census.py` scans all git-tracked Python files outside `experiments/`, parses
their ASTs, resolves import aliases, and records non-decode operator calls
and function-valued references. It found **zero parse failures**. It is a
static source census, not a claim that every parameter combination executes
successfully or that every runtime-selected dtype is statically known.

The initial census has:

- 234 TTNN sites in 113 files: 138 sites under models (134 calls, four function
  references) and 96 calls under tests. Model-local tests are included under
  models. Two test calls are the prior research repro and are separately
  excluded from production compatibility obligations in the tests report.
- 76 Torch-reference sites, not TTNN migration callers.
- 14 TTML sites described above.
- 38 indirect/name-only occurrences manually reviewed in the model reports:
  mostly DeepSeek B1 data variables, plus wrapper/monkeypatch/reference cases.

Function-valued dispatch sites require manual tracing; the companion reports
perform that work and resolve precision/input provenance or explicitly mark
runtime-dependent values. Text/C++ searches supplement Python AST coverage.
Submodule implementations, external consumers, and untracked experiment
snapshots are not counted as production callers. The audit is tied to the
current checkout, not a freshly fetched upstream revision.
