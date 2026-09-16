# Kernel Lib Pipeline Watcher

Third sibling of the SDPA (`../.sdpa-watch/`) and Conv (`../.conv-watch/`) pipeline
watchers — **same infrastructure**, different scope and channel. Monitors
`tenstorrent/tt-metal` GitHub Actions workflows on `main` for failures in the
**ttnn kernel library** (`ttnn/cpp/ttnn/kernel_lib`) and posts an hourly digest
to a dedicated Slack channel.

Scope = the kernel library's own test suite, `tests/ttnn/unit_tests/kernel_lib`:
the eltwise chain suites (`test_chain_blocking`, `_dest_accumulation`,
`_elements`, `_indexing`, `_lifecycle`, `_oob`, `_optional`, `_perf`,
`_reconfig`, `_skip_compute`) plus `reduce/test_reduce_helpers.py`. It does
**not** cover the ops that consume the library, and it does **not** cover the
tt-llk repo's own smoke jobs (`llk-sanity-tests`, `llk-build-quasar`) — those
are the LLK team's.

## What it watches

Five independently tracked configurations in `config.sh` `PIPELINES`:

| State key | Workflow/event | In-scope jobs |
|---|---|---|
| `sanity-push` | `sanity-tests.yaml`, push | `kernel lib tests [wh_n300_civ2]` |
| `sanity-scheduled-bh` | `sanity-tests.yaml`, schedule | `kernel lib tests [bh_p150b_civ2]` |
| `debug-plain` | `sanity-tests-debug.yaml`, 00:00 schedule | Ubuntu 22.04/24.04 × WH/BH |
| `debug-watcher` | `sanity-tests-debug.yaml`, 01:00 schedule | same four jobs, watcher enabled |
| `debug-llk-asserts` | `sanity-tests-debug.yaml`, 02:00 schedule | same four jobs, LLK asserts enabled |

Adding a group later is a one-line `PIPELINES` edit; the next tick picks it up
with no restart.

### Two group spellings, on purpose

The job pattern is `(ttnn llk helper library|kernel lib) tests`. The current
name on `main` is **`kernel lib tests`**. The old spelling remains in the
matcher so historical runs can still be inspected with `dryrun.sh`.

Job names carry the SKU — `prepare_test_matrix.py` appends `[<sku>]`
unconditionally — so the real strings look like
`ttnn-sanity-tests / kernel lib tests [wh_n300_civ2]`.

### SKU and variant coverage

- **Sanity push** runs `wh_n300_civ2` (Wormhole N300). A separate sanity cron
  every two hours runs `bh_p150b_civ2` (Blackhole P150b). They have separate
  state keys, so a busy push queue cannot hide a completed BH scheduled run.
- **Debug Sanity** has **three scheduled variants sharing one workflow file**:
  plain RelWithDebInfo (00:00 UTC), `with watcher` (01:00 UTC), and `with LLK
  asserts` (02:00 UTC), each across Ubuntu 22.04 + 24.04 and both SKUs. Each
  variant has its own state key and run-title selector. The stale-page guard
  therefore compares run numbers only within the same configuration. A slow
  watcher run remains visible even when the later LLK-assert run finishes first.

### Debug Sanity is pinned to `event=schedule`

Each debug entry selects `event=schedule`, so a manual `workflow_dispatch` run
cannot flip the scheduled configuration's digest. A fix-attempt dispatch is
invisible to the hourly digest; inspect it directly instead:

```bash
~/.kernel-lib-watch/dryrun.sh sanity-tests-debug.yaml <run_id>
```

## Runtime vs snapshot (read `../.sdpa-watch/README.md` + `SETUP.md` first)

- **Runtime** = `~/.kernel-lib-watch/` — the cron job, cache, secrets, logs. Source of truth.
- **Repo snapshot** = this dir — a secrets-free copy of the functional files for
  review/porting. Only `config.sh`, `watch.sh`, `agent_prompt.txt`,
  `ensure-cron.sh`, `dryrun.sh` are tracked; `slack_webhook`, `slack_bot_token`,
  `oauth_token`, `state.json`, `.watch.lock` and `*.log` are runtime-only and
  never committed.

## Differences from the sibling watchers

- Digest title is "Kernel Lib Pipelines"; `agent_prompt.txt` scopes triage to
  the kernel library and adds two rules the siblings don't have: name the debug
  variant, and never describe unreported tests as passing (the suite runs under
  `pytest -x`, so it stops at the first failure).
- Cron fires **hourly at :15** — sdpa is :00, conv is :30 — so the three never
  race on the shared `~/.claude/.credentials.json` OAuth refresh.
  `ensure-cron.sh` MARKER/CRON_LINE point at `.kernel-lib-watch`.
- The scripts are **self-locating** (`SDPA_HOME` from `BASH_SOURCE`), so this
  directory reads its own `~/.kernel-lib-watch/` config/state, never sdpa's or
  conv's.
  `SDPA_HOME` deliberately keeps its original name: renaming it would turn
  every future fix-port between the three copies into a manual merge.

## Before the first real tick

Two things are **not** configured in this snapshot and must be supplied per host:

1. **`SLACK_CHANNEL_ID` in `config.sh` is intentionally empty.** Fill it in
   (Slack → channel → About → the `C0…` ID) and drop an xoxb token with
   `chat:write` into `~/.kernel-lib-watch/slack_bot_token`, then `/invite` the bot to
   the channel. `DRY_RUN=1` needs none of this.

   Note the channel ID only matters in **bot mode** — it is what the
   `chat.postMessage`/`chat.update` payload names. In **webhook mode** the
   destination is baked into the webhook URL itself and `SLACK_CHANNEL_ID` is
   ignored entirely. So with no `slack_bot_token` and no `slack_webhook`
   present, a real (non-`DRY_RUN`) tick exits 1 — but dropping in a
   `slack_webhook` makes it post to wherever *that URL* points, empty channel
   ID or not. Don't reuse the sdpa or conv webhook file here.
2. **Set `TT_METAL_DIR` in the host-local `config.sh` or environment to a clean
   clone that tracks `main`.** The repo snapshot deliberately leaves it empty
   so it does not record a username or machine-specific checkout path. It is only
   used for commit-range lookups, but the agent also greps it when hunting a
   likely cause — so a clone carrying unpushed local branches lets it cite a
   SHA that does not exist upstream. Observed during validation: pointed at a
   working clone, the agent correctly diagnosed the failure but named a local
   unmerged fix commit in the digest. Harmless in a personal channel,
   confusing in a shared one.

## Setup on a new host

Follow `../.sdpa-watch/SETUP.md`, but: copy these files into `~/.kernel-lib-watch/`,
set up the Slack channel/token per the section above, add the `~/.bashrc` hook
for `~/.kernel-lib-watch/ensure-cron.sh`, and install the `15 * * * *` crontab line.
Auth (`gh`, Claude credential) is shared with the sibling watchers — nothing
extra to configure.

```bash
mkdir -p ~/.kernel-lib-watch && chmod 700 ~/.kernel-lib-watch
cp .kernel-lib-watch/{config.sh,watch.sh,agent_prompt.txt,ensure-cron.sh,dryrun.sh,README.md} ~/.kernel-lib-watch/
chmod +x ~/.kernel-lib-watch/{watch.sh,ensure-cron.sh,dryrun.sh}
echo '{}' > ~/.kernel-lib-watch/state.json
# edit ~/.kernel-lib-watch/config.sh: SLACK_CHANNEL_ID and host-local TT_METAL_DIR
DRY_RUN=1 ~/.kernel-lib-watch/watch.sh          # verify without posting
~/.kernel-lib-watch/ensure-cron.sh              # installs cron + the :15 crontab line
```

After changing triage extraction or the Claude prompt, force one cached
configuration through analysis again with, for example,
`REANALYZE_CONFIG=debug-watcher FORCE=1 ~/.kernel-lib-watch/watch.sh`.

`ensure-cron.sh` needs the `cron` package, a running daemon, and `crontab(1)`;
it apt-installs and starts them itself given passwordless sudo + network. On a
host with none of that present, it self-repairs on the first login shell — see
"Reboot durability" in `../.sdpa-watch/SETUP.md`.

## Validation performed

- `bash -n` clean on all four scripts; `config.sh` sources and all `PIPELINES`
  entries split into the right 7 fields (the regex `|` lands safely in the
  trailing field).
- Job pattern checked against real API job lists: 1/57 jobs matched on sanity
  run `34957007390`; 4/10 *failed* jobs matched on debug run `34920175153`;
  `llk-sanity-tests`, `llk-build-quasar` and the LLK docker build correctly
  excluded; the post-rename spelling matches.
- `dryrun.sh sanity-tests-debug.yaml 34920175153` and a full
  `DRY_RUN=1 watch.sh` both produced correct digests end to end — Sanity
  collapsed to `✅`, Debug Sanity reported the real `test_copy_dest_int32`
  int32-CopyDest hang on both SKUs under `with LLK asserts`, with an accurate
  likely-cause commit that is genuinely on `main`.
- On 2026-09-16, selectors resolved the latest run independently for all five
  configurations. In particular, `debug-watcher` selected run `35043348869`
  while `debug-llk-asserts` selected the newer run `35046937815`, proving the
  stale-run comparison no longer crosses configurations.
