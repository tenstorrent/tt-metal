# Conv Pipeline Watcher

Sibling of the SDPA pipeline watcher (`../.sdpa-watch/`) — **same infrastructure**,
different scope and channel. Monitors `tenstorrent/tt-metal` GitHub Actions
workflows on `main` for failures in the ops/models **owned by
`@tenstorrent/metalium-developers-convolutions`** (per `.github/CODEOWNERS`) and
posts a digest to a dedicated Slack channel (`#conv-watch`).

Scope = the conv family (conv2d/1d/3d, conv_transpose2d, maxpool/avgpool/global/
adaptive pool, upsample, grid_sample, fold, sliding_window, cnn/tt_cnn) plus the
two conv-team-owned models **resnet50** and **stable diffusion 1.4**.

> **2026-08-21 — edit-in-place digest (Slack bot API), same as sdpa-watch.**
> With `~/.conv-watch/slack_bot_token` (xoxb, scope `chat:write`) and
> `SLACK_CHANNEL_ID` set in `config.sh`, the watcher edits its standing digest
> message in place while the status is unchanged (tick times accumulate on a
> `checked:` line, last 48 kept) and posts a new message only on a status
> change. State in `state.json` under `_slack`. The `sdpawatch` bot must be a
> member of `#conv-watch` (`/invite @sdpawatch`). Missing token/channel ID →
> legacy webhook behavior (new message per tick). Full description in
> `../.sdpa-watch/README.md`.

## Runtime vs snapshot (read `../.sdpa-watch/README.md` + `SETUP.md` first)

- **Runtime** = `~/.conv-watch/` — the cron job, cache, secrets, logs. Source of truth.
- **Repo snapshot** = this dir — a secrets-free copy of the functional files for
  review/porting. Only `config.sh`, `watch.sh`, `agent_prompt.txt`,
  `ensure-cron.sh`, `dryrun.sh` are tracked; `slack_webhook`, `oauth_token`, `state.json`,
  and `*.log` are runtime-only and never committed.

## Differences from the SDPA watcher

- `watch.sh` is **self-locating** (`SDPA_HOME` derived from `BASH_SOURCE`), so the
  dir is a clean copy — it reads its own `~/.conv-watch/` config/state, never sdpa's.
- Digest title is "Conv Pipelines"; `agent_prompt.txt` scopes triage to conv/pool.
- `dryrun.sh <workflow.yaml> <run_id>` previews one run's digest block without
  posting. Also self-locating, so it reads this dir's config, never sdpa's.
- Cron fires at **:30** (sdpa is :00) to avoid the shared `~/.claude/.credentials.json`
  OAuth-refresh race. `ensure-cron.sh` MARKER/CRON_LINE point at `.conv-watch`.
- Watches **14 pipelines** (see `config.sh` `PIPELINES`): 2 sanity gates (Sanity,
  Debug Sanity), Merge Gate C++ smoke, L2 Nightly conv/pool +
  tt-cnn + di/dt, Device Perf, Frequent Models (resnet50 + SD1.4 PCC), Single-card
  Demos, Model Perf, 4× T3K, Galaxy, and conv/pool Sweeps.

  **2026-08-14:** `blackhole-sanity-tests.yaml` was retired upstream (PR #48943) and
  folded into `sanity-tests.yaml`, so the separate *Blackhole Sanity* entry was dropped;
  *Sanity* now matches `ttnn conv group [sku]` / `ttnn pool group [sku]` across the
  Wormhole, Blackhole and ttsim SKUs in one workflow. 15 → 14 pipelines, no loss of coverage.

## Setup on a new host

Follow `../.sdpa-watch/SETUP.md`, but: copy these files into `~/.conv-watch/`,
create a `#conv-watch` Slack incoming webhook → `~/.conv-watch/slack_webhook`,
add the `~/.bashrc` hook for `~/.conv-watch/ensure-cron.sh`, and install the
`30 * * * *` crontab line. Auth (`gh`, Claude credential) is shared with the
sdpa watcher — nothing extra to configure.
