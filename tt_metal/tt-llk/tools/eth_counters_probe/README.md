# Ethernet counter probes

Read-only scripts; see `docs/performance_counters/eth_counters.md` section 7. Each states in its docstring what it must show.

- `eth_idle_drift.py`: single card, nothing running. Queue registers readable and monotonic, idle rate, refresh of the firmware L1 block, firmware queue copies against the registers.
- `eth_mib_config.py`: Blackhole, reads the two MAC configuration registers that decide whether the MAC statistics counters can be read without disturbing the firmware.
- `eth_validate.py`: any multi-chip system, slow dispatch, the all-connected-chips ethernet unit test with `TT_METAL_PROFILE_ETH_COUNTERS` off and on.

Run each only on an idle system nothing else has open. The scripts never write a device register; `eth_validate.py` runs a unit test that writes the L1 of its own ethernet cores like any test.
