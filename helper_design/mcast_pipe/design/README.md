# Design evidence

These documents explain why the current helper API has its present shape. They
are still authoritative design evidence, but they are not the rollout's daily
working state.

- `primitive_contracts.md` records the low-level NoC and semaphore contracts.
- `hazards_catalog.md` records races, invariants, and allowed mitigations.
- `api_feasibility.md` checks the API against the production call-site inventory.
- `style_bakeoff.md` records the device experiments behind implementation choices.

Read them when changing the helper API, investigating a capability gap, or
checking whether a migration preserves the original synchronization contract.
For current status and next work, start at `../README.md` and
`../migration/ledger.json`.
