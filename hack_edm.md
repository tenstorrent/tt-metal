# Fabric EDM Hacks

Two independent changes to the fabric EDM layer for all-gather perf work.
NOTE: this works on Wormhole but hangs on Blackhole (likely because BH EDM has two separate
Baby RISCs that are currently pinned to specific NOCs).

---

## Hack 1: Change `edm_to_local_chip_noc` from 1 to 0

Move the EDM's local-chip write NOC from NOC 1 to NOC 0. This requires updating the device-side constant AND all host-side values that feed the compile-time arg (a static_assert enforces they match).

### Files and changes

**`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp`**
- Change `edm_to_local_chip_noc` from `1` to `0`

**`tt_metal/fabric/erisc_datamover_builder.hpp`**
- Change `DEFAULT_RECEIVER_LOCAL_WRITE_NOC` from `1` to `0`
- Change `BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_LOCAL_WRITE_NOC` from `1` to `0`

**`tt_metal/fabric/builder/fabric_core_placement.cpp`**
- In `run_default_galaxy_optimizer`, both `if` branches set `receiver_channel_local_write_noc_ids[i]` to `1` for both builders. Change all four assignments from `1` to `0`.

**`tt_metal/fabric/fabric_tensix_builder_impl.hpp`**
- In `enum class UdmNoCSelection`, change `relay_noc` from `1` to `0` (comment says it must match `edm_to_local_chip_noc`).

---

## Hack 2: Separate credit/seminc writes onto a distinct NOC virtual channel

Put `noc_semaphore_inc` (credit/completion signals) on VC 0, independent from data writes which use VC 2/3 (`forward_and_local_write_noc_vc`). This eliminates head-of-line blocking at NOC routers where small credit messages get stuck behind large data packets. Callers that need data-before-seminc ordering must set `flush=true` in the packet header.

### Files and changes

**`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp`**
- Add new constant: `static constexpr uint8_t edm_credit_noc_vc = 0;`

**`tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp`**
- In `execute_chip_unicast_to_local_chip_impl`, change the VC parameter of all 3 `noc_semaphore_inc<true>` calls from `tt::tt_fabric::forward_and_local_write_noc_vc` to `tt::tt_fabric::edm_credit_noc_vc`:
  1. `NOC_UNICAST_ATOMIC_INC` case (standalone seminc)
  2. `NOC_FUSED_UNICAST_ATOMIC_INC` case (data + seminc)
  3. `NOC_UNICAST_SCATTER_WRITE` case (scatter chunks + final seminc)
- Do NOT change `noc_async_write_one_packet_with_trid` (data writes) or `noc_inline_dw_write` (inline writes) -- those stay on the data VC.
