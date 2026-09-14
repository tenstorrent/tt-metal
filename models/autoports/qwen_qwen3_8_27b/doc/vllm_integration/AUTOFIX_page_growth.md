# AutoFix: adapter page-growth contract

## Starting evidence

`AUTODEBUG_page_growth.md` documents source inspection of vLLM
`03fa3af2e15b5f8dc07cbaa67d92f979aa00be11`. The runner updates and forwards
allocator page tables during steady decode without asserting a full input reset.
The existing `QwenGenerator._refresh_table` already supports page-only refreshes.
The investigation refuted the stronger claim that absent runner layout
invalidation alone necessarily makes the device table stale.

The parent agent subsequently authored `tt/generator_vllm.py`; this test pass
inspected that adapter and authored only
`tests/test_vllm_adapter_host.py` and this report. No implementation was changed
by the testing agent, and no device or server was opened.

## Hypothesis experiments

### Page growth preserves device tokens and positions

**Hypothesis:** With `reset_batch=False` and decode already bound, the adapter
forwards a changed normalized page table but passes `tokens=None` and
`start_pos=None` to the generator. The canonical table comparison produces one
in-place page-table copy and no token/position refresh.

**Experiment:** Invoke the actual `Qwen38ForCausalLM.decode_forward` using a fake
generator whose `_refresh_table` is the real `QwenGenerator._refresh_table`.
Replace only device copy, replay, sampling setup, and recurrent remap effects
with observable host operations. Initialize device token/position state to values
different from the intentionally stale host inputs. Add physical block 23 to the
first request's table while retaining the second request's mapping.

**Result:** The new table reached the existing table buffer; tokens and positions
retained their previous values; the only refresh was `page_table_refreshes=1`.
The fake output object was returned without triggering readback. **Verified at
the host adapter boundary.**

### Unchanged table values do not cause per-token copies

**Experiment:** Submit three separately cloned but value-identical tables;
normalize a narrower table whose zero padding matches the bound table; mutate a
caller's original table after a submission and submit it again.

**Result:** Equal tables caused zero copies. The caller mutation did not alter
the saved snapshot and caused exactly one subsequent page refresh. The actual
generator validator also rejected an ID equal to `cache.num_pages` before replay.
**Verified.** These adapter tests receive complete tables, not scheduler allocation
payloads; equivalence of empty block lists and unchanged persistent tables is
documented from upstream source in the AutoDebug report.

### Reset, cache identity, and remap sequencing

**Experiment:** Exercise first decode with `reset_batch=False`, a later explicit
reset with an inactive row, incompatible cache objects on either side of the
adapter/generator binding, and a slot permutation combined with an explicit
reset and correspondingly reordered host inputs/table.

**Result:** Initial binding and explicit reset supplied authoritative host inputs;
the reset recomputed active slots from nonnegative positions. Incompatible cache
objects failed before remap, sampling, or replay. The remap event preceded sampling
setup, table/token/position refresh, and decode. **Verified at the adapter
boundary.** The adapter receives already prepared reset inputs; these tests do
not independently execute the runner's pending-output drain.

### Negative controls

The tests also ran against two temporary, in-memory replacements of the actual
adapter method, using `inspect.getsource`, `compile`, and
`unittest.mock.patch.object`; no implementation file was modified:

1. Replace `table = self._table(page_table)` with
   `table = self._table(page_table) if refresh else self.generator.page_host.clone()`.
2. Replace `refresh = reset_batch or not self._decode_bound` with `refresh = True`.

For each replacement, run
`AdapterHostTests('test_growth_refreshes_pages_and_preserves_device_token_position')`
into a `unittest.TestResult` and require exactly one assertion failure and zero
errors. Both replacements were rejected as expected. The first proves the test
detects lost allocator mapping; the second proves it detects stale host-input
reload. These are expected negative-control failures, not failures of the final
implementation.

## Verification commands and results

From the tt-metal repository root:

```bash
source ../run-env.sh
PYTHONPATH="$VLLM_ROOT:$PYTHONPATH" python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.test_vllm_adapter_host
python_env/bin/python -m black --check models/autoports/qwen_qwen3_8_27b/tests/test_vllm_adapter_host.py
```

Result on 2026-09-14: **10 tests passed in 0.005 seconds**. Black reported that the
file would be unchanged and exited successfully. It emitted an environment
warning about Python 3.12 versus its inferred Python 3.14 target; no formatting
changes were required. The imports loaded the installed TTNN bindings, but no
mesh, model, weights, device tensor, or device operation was created.

No build is required for these Python tests and markdown reports.

## Final status and limits

The proposed adapter page-growth handling is verified by focused host tests,
including negative controls for both competing failure mechanisms. There is no
failing host result to send back for an implementation repair.

The allocator-driven runtime control in `AUTODEBUG_page_growth.md` remains
required: actual scheduler allocation events, page-only copies at growth,
unchanged device token/position state, exact output IDs across boundaries, and
pending-output completion before genuine resets. This pass does not verify
device queue ordering, per-submission output lifetime, full-model numerics,
attention rounded-window allocation coverage, or runtime slot movement. It must
not be cited as a hardware serving pass.
