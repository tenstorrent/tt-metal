# Proposed selected-policy defaults

[The unapplied patch](selected_policy.patch) makes the same policy the default in `MultichipDecoder.from_state_dict`, the headline runner, the stack test and the batch/prefix contract test. It is a proposal pending attention-DRAM results and final stress gates, not stage acceptance. No live source or device was modified during preparation.

| Setting | Constructor and harness default | Explicit comparison override |
| --- | --- | --- |
| Hybrid experts | enabled | `--no-hybrid-experts` |
| Fused tail | enabled | `--no-fused-tail` |
| Optimized shared decode | enabled | `--no-optimized-shared --shared-geometry 0` |
| Grouped shared/routed reduction | enabled | `--no-grouped-moe-reduce` |
| Shared decode geometry | 1 | `--shared-geometry 0` or `2` |
| QKV and WO decode fidelity | LoFi for both layer kinds | `--qkv-fidelity` / `--output-fidelity` |
| Attention CCL dtype | BF16 for both layer kinds | `--attention-ccl-dtype float32` / `bfloat8_b` |
| Full-attention-only CCL override | none; inherits BF16 | `--full-attention-ccl-dtype bfloat8_b` |
| Residual/topology | replicated / Linear | existing explicit sharded/Ring controls |
| Shared/attention DRAM, fused AGMM | disabled | existing headline-runner candidate flags |

The four selected booleans previously used `store_true`, producing explicit False values that overrode the constructor. All three harnesses now use `argparse.BooleanOptionalAction(default=True)`, retaining the positive flags while adding `--no-*`. Shared geometry, grouped reduction and projection/CCL choices are forwarded by stack/contracts as well as the headline runner. Fidelity and dtype fields are recorded in each result.

Projection fidelity is configured in the constructor instead of changing objects after construction in the headline runner. `_Projection` receives the selected decode fidelity while its cached prefill projection retains the original compute config. WO passes the selected fidelity to its existing constructor and keeps the configured prefill path. FP32 destination accumulation, approximate math and packer settings are unchanged. This also keeps the attention-DRAM helper compatible because it reads the current compute policy on each call.

`attention.reduce` now calls a model method that casts to the selected attention CCL dtype before the existing all-reduce/reduce-scatter callback. It matches the earlier runner-only cast boundary for both prefill and decode; shared and routed branches are unaffected. Full-only override resolution happens inside the model constructor using the actual layer kind. BF8 full remains an explicit test override until stack and stress evidence supports selecting it. The default patch does not select BF8 sliding or full.

Unselected False settings were audited: `expert_parallel`, `sharded_residual`, `fused_agmm` and `shared_dram` remain False intentionally; `attention_dram` remains None. EP-only experiments now require `--expert-parallel --no-hybrid-experts`. Sharded-residual experiments require `--sharded-residual --no-grouped-moe-reduce`. Shared DRAM still requires geometry0. Parser errors identify these combinations before hardware setup. There is no silent policy fallback. Measurement-only options such as `--trace`, `--check-cache`, profiling and reservation flags retain their existing explicit semantics; headline evidence must still pass `--length 4096 --steps 128 --trace --check-cache`.

## Exact bases and checks

[Provenance JSON](selected_policy_provenance.json) contains candidate hashes and full proposed snapshots. The patch is based on the integrated attention-DRAM candidates:

| File | Base SHA256 |
| --- | --- |
| `tt/multichip_decoder.py` | `49cd4b3e47aeca24389fd58c29262556d05ec35cf0f0d80c8077c02af51f88e5` |
| `tests/run_multichip_decoder.py` | `acf7c7991775248983d779bc6e782d6f803516528c998213d01aeb4a05138876` |
| `tests/test_multichip_stack.py` | `ef3a5f297b197ea1c70eb7e5d97860f131ec53cd257381065d74d5b5de15c81b` |
| `tests/test_multichip_contracts.py` | `4ff91b32bf1bdd761e6199425217995c03b6e88281b5c67efaa9704b414bf409` |

All four complete candidate files parse. `git apply --check` passes against those bases. [CPU checks](selected_policy_cpu_checks.json), with [preserved check source](selected_policy_cpu_checks.py.txt), execute only extracted argparse setup and small policy adapters; they import no TTNN. They verify identical defaults across all three harnesses, explicit disabling, candidate overrides, full-only BF8 resolution, invalid-combination rejection and cast-before-reduce placement. They do not execute tensor math or establish hardware correctness. Defaults need fresh headline, stack, batch/prefix, capacity, profile and Watcher evidence before final selection/acceptance.
