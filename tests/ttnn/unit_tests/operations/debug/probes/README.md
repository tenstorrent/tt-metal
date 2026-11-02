# Probes that are not pytest tests

These are kept so nothing from the Quasar ProgramSpec bring-up is lost. They are not collected by pytest. The other
probes from the same work became tests in the parent directory, such as `test_generic_spec_op_trisc_l1_read.py` and
`test_generic_spec_op_dfb_multi_entry_lock.py`.

| File | What | Result |
|---|---|---|
| `quasar_usable_l1.c` | Computes Quasar's per-core L1 budget from `dev_mem_map.h`; the build command is in the file | Unreserved base 0x51640, usable 3,860,928 B (3.68 MiB) per core, about 123.5 MB over 32 cores |
| `blaze_conftest_shim.py` | Copy it as `conftest.py` next to a copy of a tt-blaze test to run that test outside the blaze tree; it loads blaze's root and `tests/blaze` conftests | Used for the 2x1 SDPA probe below |

## 2x1 SDPA "hang" on craq-sim (2026-11-02)

- **Probe:** blaze's `tests/blaze/micro_ops/common/test_sdpa_metal2.py::TestSDPAQuasar::test_sdpa[quasar-127-2x1-mesh_device0]`, run with the shim above. Blaze has un-skipped this case, so its own test is now the reproducer and no copy is kept here.
- **Finding:** watcher-induced simulator clock starvation, not a deadlock.
  - UMD clocks craq-sim only from host reads, one cycle per read (`TTSimTTDevice::after_read`).
  - Slow-dispatch `wait_until_cores_done` (`tt_metal/llrt/llrt.cpp`) sleeps 5 ms per poll round when the watcher is on. With 2 cores that is about 400 sim cycles/s.
- **Results** (craq-sim md5 `40f833f07975ea18ad2310dcd58a644c`):
  - `TT_METAL_WATCHER=1` looks hung.
  - Watcher off: passes in 12 s.
  - `TT_METAL_WATCHER=10ms` + LLK asserts: passes in 82 s.
- **Rule:** on craq-sim, use `TT_METAL_WATCHER=10ms`.
