# LLK formal contract

Formal contract for the LLK (Low-Level Kernel) software layer, plus supporting material.

- [`llk-formal-contract.md`](./llk-formal-contract.md): the math contract. The words we use, how
  the state splits, what the entry points and contract functions do, and the per-EXU lifecycle FSM.
- [`san_contract_gaps.md`](./san_contract_gaps.md): the companion. Where today's Sanitizer and LLK
  code do not match the contract (bugs, rules not switched on, coverage gaps, audits, next steps).
- [`related-work.md`](./related-work.md): an annotated bibliography. §1-§4 on tying API/function to
  software and hardware state (typestate, ISA as spec, hardware-software contracts, register
  standards); §5-§9 on the prior art each planned paper must cite and position against.
- [`papers-plan.md`](./papers-plan.md): the plan for three conference papers (the LLK contract,
  the HAL, and static state tracking) and how they build on each other.
