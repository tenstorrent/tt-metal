# Expectations: Hardware / Firmware / Kernels

> **Codeowners:** `@abhullar-tt @jbaumanTT @nathan-TT @kstevensTT` (firmware/hw general);
>   `@rtawfik01 @rdjogoTT @nvelickovicTT @amahmudTT @lpremovicTT @ncvetkovicTT @fvranicTT` (ckernels/LLK);
>   `@aliuTT @ubcheema` (Ethernet firmware)
> **Paths:** `tt_metal/hw/`
> **Status:** AI-generated draft — codeowners please review and correct

`tt_metal/hw/` contains firmware, hardware-specific includes, compute kernels (LLK/SFPU), and toolchain support. This code runs directly on Tenstorrent hardware — bugs here can cause silent data corruption, hangs, or hardware damage.

---

## Hard Blockers

- [ ] **Firmware changes must not regress existing dispatch or Ethernet behavior.**
  Changes to firmware under `tt_metal/hw/firmware/` interact directly with the dispatch pipeline. Any behavioral change must be described explicitly and validated on real hardware.

- [ ] **No changes to kernel API headers without coordinating with LLK owners.**
  `tt_metal/hw/inc/api/compute/compute_kernel_api.h` and related kernel API headers are the interface between the host and device kernels. Changes require explicit sign-off from the ckernels/LLK codeowners (`@rtawfik01` et al).

- [ ] **Ethernet firmware changes require both Ethernet and hardware owners.**
  `tt_metal/hw/firmware/src/tt-1xx/*erisc*` is co-owned by `@aliuTT @ubcheema`. Changes here need both hardware and Ethernet firmware review.

- [ ] **L1 address map changes must be synchronized with host-side consumers.**
  Changes to `tt_metal/hw/inc/internal/*/eth_l1_address_map.h` or other address map headers affect host/device shared memory layout. These require coordinated changes on both the firmware and host sides.

---

## Guidance

- **LLK/SFPU changes are numerically sensitive.** SFPU operations underpin all on-device math. Changes here need careful numerical validation — compare outputs against reference implementations.

- **RISC-V code size is a hard constraint.** Firmware images have fixed binary size limits. Adding code may require removing or shrinking something else.

- **`tt_metal/hw/inc/` is a public/internal boundary.** Headers under `tt_metal/hw/inc/api/` are intended for kernel authors and should be documented and stable. Headers under `tt_metal/hw/inc/internal/` are implementation details.

- **Hardware-specific code must be clearly guarded.** If code only applies to Wormhole or Blackhole, use the appropriate compile-time guards. Don't let WH-only logic silently run on BH or vice versa.

---

## Common Feedback

- _"This changes a shared header — did you validate on both WH and BH?"_
- _"LLK output doesn't match reference for edge cases (NaN/Inf/subnormal)."_
- _"Firmware binary size check — does this fit?"_
- _"This L1 address map change needs a matching host-side update."_

---

## Testing Requirements

- [ ] Firmware changes must be validated on real WH/BH hardware.
- [ ] LLK/SFPU changes must include sweep tests covering numerical edge cases (zero, NaN, Inf, subnormal, min/max).
- [ ] Compute kernel API changes must not break existing kernel tests.

---

## Notes for External Contributors

Hardware/firmware code in tt-metal is as close to bare metal as it gets. If you're not familiar with the RISC-V firmware architecture or LLK programming model, coordinate with the codeowners before making changes — the failure modes are subtle and often only visible on real hardware.
