# Design evidence

These documents explain why the current helper API has its present shape. They
are still authoritative design evidence, but they are not the rollout's daily
working state.

The feasibility and bake-off narratives reached their semantic conclusions at
API v11. API v12-v14 changed argument-wire ownership and call-site spelling,
not the multicast protocol decisions recorded here. The materialized v14
headers and `../migration/ledger.json` are authoritative for current API and
rollout status.

- `primitive_contracts.md` records the low-level NoC and semaphore contracts.
- `hazards_catalog.md` records races, invariants, and allowed mitigations.
- `api_feasibility.md` checks the API against the production call-site inventory.
- `style_bakeoff.md` records the device experiments behind implementation choices.

Read them when changing the helper API, investigating a capability gap, or
checking whether a migration preserves the original synchronization contract.
For current status and next work, start at `../README.md` and
`../migration/ledger.json`.

## Active extension plan

- [`../plan.md`](../plan.md) defines the agreed
  `McastFamily`/`McastGroup`, exact multi-rectangle, GroupNorm,
  chain-forwarding, and Conv3D implementation sequence. Its execution state is
  tracked in [`../tracker.md`](../tracker.md).
