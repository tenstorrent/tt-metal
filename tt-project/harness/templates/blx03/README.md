# blx03 legacy job scripts

Shared blx03 tooling that lived in `tmp/` on ttp/t48-ltx25-integrated (#150). Originals stay
on `ttp/t48-notes-archive` @ b81bb403d86. New device work should use the serial runner
(`../blx03-runner/`); these are for the per-task driver path (`../blx03-launch.sh`) and as
`CMD`/`ENV` targets of a runner spec.

| here (g15blx02) | live on blx03 | what |
|---|---|---|
| `submit.sh` | `~/fasth3/tt-metal/tmp/blx03/submit.sh` | queue one job; refuses while a smarton job is active (75) or on a bare 2x4 mesh open (64) |
| `env.yaml` | `~/fasth3/tt-metal/tmp/blx03/env.yaml` | broker env file (`-e`, or a runner spec's `ENV`) |
| `run25.sh` | `~/fasth3/tt-metal/tmp/blx03/run25.sh` (local edits, see below) | LTX-2.5 1080p e2e on the shared tree |
| `run48.sh` | `~/fasth3/t48/tmp/blx03/run48.sh` | t48 e2e on its own worktree and build |
| `driver.sh` | per task, `~/fasth3/t<n>drv/driver.sh` | per-task detached driver template (t78 example) |
| `test/test_health.sh` | - | offline check of `driver.sh`'s health parsing |

Sources: `submit.sh` from ttp/t36-blx03-ltx25 @ 86076afc66 (the blx03 copy; t48's is older and
lacks the duplicate-submit and bare-2x4 guards). The rest from t48 @ b81bb403d86.

blx03's `~/fasth3/tt-metal` (t36 checkout) is the live copy that running drivers call. Leave it
as it is. Its `run25.sh` carries an uncommitted 1150 MHz `hostfmax.py` clamp and defaults to
the DiffVAE; the copy here is t48's (conv decoder default, no clamp).

## Overlap with the runner

`submit.sh` and `driver.sh`'s `health()`/`run_job()` do what `../blx03-runner/runner.sh` does
(busy check, health gate, drop detection). The runner is the one to extend; do not add features
here. Both refuse while the other's job is active (see the runner README's deploy note).
