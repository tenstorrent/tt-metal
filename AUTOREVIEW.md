# Auto review: branch fplavec/gpt_oss_quasar_bringup

- Target: all commits on the branch since it left `main`
- BASE (merge-base with `origin/main`): `72c529c7e702a9a1632e724d9d652a4aba9f7cd3`
- BASELINE (branch head, fixes applied on top): `6c6d739d4aaec88b91079e544e0adf59f3622a9a`
- Date: 2026-11-04

## 1. Summary per reviewer

| Reviewer | Reviews | Result |
|---|---|---|
| 1 | 2 of 5 | NO FURTHER COMMENTS |
| 2 | 3 of 4 | NO FURTHER COMMENTS |
| 3 | 2 of 3 | NO FURTHER COMMENTS |

## 2. All issues raised

### Reviewer 1

- **R1-F1.** The gpt-oss copy of `determine_device_name` gave Quasar its own name "QSR". The shared helpers at BASE give Quasar the Blackhole names ([device_utils.py:25-27](models/common/device_utils.py#L25-L27)).
  **Fixed:** Quasar now takes the Blackhole names, with the same comment and TODO ([model_config.py:415-424](models/experimental/ops/quasar/gpt_oss/tt/model_config.py#L415-L424)). No sequence-length table has an entry for "P150" or "P100", so behavior is unchanged.
- **R1-F2.** `getattr(ttnn, name, None)` skipped a missing op with no message.
  **Fixed:** `getattr(ttnn, name)` with no default ([conftest.py:84-93](models/experimental/ops/quasar/gpt_oss/conftest.py#L84-L93)).
- **R1-F3.** Three wrong points in the `quasar_block_float_to_bf16` docstring: "every weight would be missing", "the ops that take an output dtype" (only the listed ones), and "1.9x more DRAM".
  **Fixed:** all three ([conftest.py:46-60](models/experimental/ops/quasar/gpt_oss/conftest.py#L46-L60)).
- **R1-F4.** "the upload runs no device conversion" was not exact: a bf16 TILE upload still tilizes on the device.
  **Fixed:** "runs neither the float32 tilize nor a device typecast" ([conftest.py:17-24](models/experimental/ops/quasar/gpt_oss/conftest.py#L17-L24)).
- **R1-F5.** The local variable `quasar_ops` had the same name as its fixture and held a module.
  **Fixed:** renamed to `quasar` ([conftest.py:116-142](models/experimental/ops/quasar/gpt_oss/conftest.py#L116-L142)).
- **R1-F6.** The `quasar_ops` docstring described details of distant code (bias handling in mainline `linear`, the arguments of the Quasar bindings). It also said that all routed mainline ops build `DataMovementKernel` kernels, which is wrong for `sparse_matmul`.
  **Fixed:** the docstring now gives only the rule ([conftest.py:107-112](models/experimental/ops/quasar/gpt_oss/conftest.py#L107-L112)).
- **R1-F7.** The `get_compute_grid_size` docstring described a before and after, and hard-coded 8x8 instead of naming the limit parameters.
  **Fixed** ([general_utils.py:17-24](models/experimental/ops/quasar/gpt_oss/utils/general_utils.py#L17-L24)).
- **R1-F8.** No test covered `get_compute_grid_size`, the Quasar device name, or the dtype mapping in `_wrap`.
  **Out of scope** for the first two: a host-only test file was written, then removed at the user's request. It is kept in `git stash` as "test_device_helpers.py (host-only tests for get_compute_grid_size and determine_device_name)".
  **Out of scope** for `_wrap`: a host test needs the wrapper builders moved out of `conftest.py` into an importable module, which is a refactor beyond these changes. The wrappers run in every Quasar run of the model.

### Reviewer 2

- **R2-F1.** In the `quasar_block_float_to_bf16` docstring, "its from_torch wrapper is the outer one" read as if the host-cast wrapper were the outer one, which is the opposite of the real order.
  **Fixed:** the docstring names both wrappers ([conftest.py:51-55](models/experimental/ops/quasar/gpt_oss/conftest.py#L51-L55)).
- **R2-F2.** The `QSR_N_LAYERS` texts give the expert cost per layer at bf4 (~448 MB). On Quasar, the model now uses bfloat16, which costs ~1.6 GB per layer.
  **Fixed** in [README.md:43-49](models/experimental/ops/quasar/gpt_oss/README.md#L43-L49) and [text_demo.py:123-129](models/experimental/ops/quasar/gpt_oss/demo/text_demo.py#L123-L129). The figure is measured: the three bfloat16 expert cache files for one layer are 530842048 bytes each.
- **R2-F3.** After R1-F6, nothing said why `linear` is routed.
  **Fixed** with a one-line comment at the route ([conftest.py:124-125](models/experimental/ops/quasar/gpt_oss/conftest.py#L124-L125)). The comment is verified: the mainline op adds some biases with `ttnn::add` ([matmul.cpp:139-170](ttnn/cpp/ttnn/operations/matmul/matmul.cpp#L139-L170), [matmul.cpp:355-365](ttnn/cpp/ttnn/operations/matmul/matmul.cpp#L355-L365)). The docstring now says "the mainline ops below". The finding assumed that the mainline `ttnn.matmul` works on Quasar. It does not in the 1D multicast config (issue #63758), and the reviewer accepted that correction.
- **R2-F4.** No test of the wrapper mapping and the wrapper order. This is the same as the out-of-scope part of R1-F8.
  **Out of scope** for the same reason. The reviewer accepted the reason.
- **R2-F5.** The new texts from R2-F2 said "see conftest.py", which was ambiguous; the README requires paths relative to the repository root.
  **Fixed:** the README names the fixture and the full path ([README.md:46-49](models/experimental/ops/quasar/gpt_oss/README.md#L46-L49)). The demo comment was changed again for R3-F3.

### Reviewer 3

- **R3-F1.** The new autouse fixtures call `is_quasar()`, and therefore `ttnn.get_arch_name()`, for host-only tests, so those tests would need a device.
  **Rejected:** the root `conftest.py` already calls `ttnn.get_arch_name()` in `pytest_addoption` for every pytest session in this repository, at BASE too ([conftest.py:983-990](conftest.py#L983-L990)). The reviewer retracted the finding.
- **R3-F2.** No test of `_wrap`, the host-cast conditions, the cache path suffix, the routing table, or the slow-dispatch trace gate. This extends R1-F8 and R2-F4.
  **Out of scope:** testing the wrappers needs the same refactor, and testing the trace gate needs the demo. The reviewer accepted the reason.
- **R3-F3.** The demo comment from R2-F5 named a fixture in another file, which goes stale if that fixture is renamed.
  **Fixed:** the demo comment keeps only "On Quasar, where the model uses bfloat16, each layer's experts cost ~1.6 GB" ([text_demo.py:126](models/experimental/ops/quasar/gpt_oss/demo/text_demo.py#L126-L126)). The README keeps the full cross-reference.

## 3. Unresolved issues

No unresolved issues.

## 4. Tests and experiments

- Host-only unit tests, the stashed `test_device_helpers.py` and [test_expert_parallel_config.py](models/experimental/ops/quasar/gpt_oss/tests/unit/test_expert_parallel_config.py#L1-L10): 16 passed, once with no Quasar settings and once with the craq-sim settings. With the craq-sim settings, `is_quasar()` was checked to be true, so the edited autouse fixtures ran too.
- The project's pinned black (`pre-commit run black`) passes on all changed files. No changed line is over 120 characters.
- gpt-oss demo on craq-sim, `QSR_N_LAYERS=2`, `test_gpt_oss_demo[quasar-1x1-prefill_128]`, with all fixes applied (log: `/localdev/fplavec/temp/gpt_oss_ops_runs/20261104_demo_autoreview_fixes.log`). The demo sets up the model, turns off trace capture under slow dispatch, and runs prefill until the attention output projection in layer 0. There, `ttnn.matmul` fails the check "Multicast sender/forwarder NoC differs from the configured multicast NoC" (issue #63758). This is the same failure point as the run at BASELINE without the fixes (`/localdev/fplavec/temp/gpt_oss_ops_runs/20261104_demo_head_6c6d739.log`), so the test fails, but not because of the fixes.
- Nothing was run on Wormhole or Blackhole hardware.

## 5. Changed files

All fixes are uncommitted edits in the working tree. Nothing was staged or committed.

Changed (`git diff --stat 6c6d739d4aa`):

- `models/experimental/ops/quasar/gpt_oss/README.md`
- `models/experimental/ops/quasar/gpt_oss/conftest.py`
- `models/experimental/ops/quasar/gpt_oss/demo/text_demo.py`
- `models/experimental/ops/quasar/gpt_oss/tt/model_config.py`
- `models/experimental/ops/quasar/gpt_oss/utils/general_utils.py`

No files were created. The new test file was moved to `git stash` (see R1-F8).
