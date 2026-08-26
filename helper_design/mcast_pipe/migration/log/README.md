# Migration logs

These are append-only per-unit implementation and validation records. Earlier
sections describe the API and status at the time they were written; later
sections may record remigration, reversion, or verification outcomes.

Use `../ledger.json` for current status and API version. A historical log is
not stale merely because it mentions an older API, but it must not claim that
an old checkpoint is the present state.
