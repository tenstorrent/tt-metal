# Cachew checkout

Use this action in place of `actions/checkout`, retaining existing checkout inputs:

```yaml
- name: Checkout
  id: checkout
  uses: tenstorrent/tt-metal/.github/actions/cachew-checkout@<full-tt-metal-commit-sha>
  with:
    ref: ${{ inputs.ref || github.sha }}
    path: docker-job
    submodules: recursive
    fetch-depth: 500
    fetch-tags: true
```

The remote reference works before the workspace exists. A local reference such as
`./.github/actions/cachew-checkout` requires a bootstrap checkout. All 20 inputs
and the `ref` and `commit` outputs from pinned `actions/checkout` v6.0.2 are
forwarded. Existing checkout conditions, IDs, paths, and timeouts still apply.

| Additional input | Default | Meaning |
| --- | --- | --- |
| `cachew` | `auto` | `auto`: self-hosted Linux runners only. `false`: never. `true`: any Linux runner |
| `cachew-url` | `auto` | `auto` probes the trusted origins; or one trusted HTTPS origin |
| `cachew-health-timeout` | `2` | Seconds for each liveness request, maximum 30 |
| `cachew-fetch-timeout` | `120` | Seconds without transfer progress before a cache fetch is abandoned, maximum 600 |

## Where cachew runs

Cachew is only reachable from runners on the CI VLAN. With the default
`cachew: auto`, the runner skips every cachew step unless
`runner.environment == 'self-hosted'` and `runner.os == 'Linux'`. On GitHub-hosted
runners (`ubuntu-latest`, `ubuntu-slim`, larger runners) the action is exactly the
pinned `actions/checkout`: no probe, no token, no Git wrapper and no configuration
change. A job or workflow `env: CACHEW_CHECKOUT: 'false'` disables cachew for every
checkout in its scope without editing each step.

The trusted origins are listed in [`trusted_caches.json`](trusted_caches.json),
one per cluster, as deployed by tt-flux `apps/cachew`:
`cachew.ci.tenstorrent.net` (Austin), `cachew.yyz2.ci.tenstorrent.net` (Toronto)
and `cachew.dev.tenstorrent.net` (dev). Restricted DNS on the runners resolves all
three names to the local cluster's Service, and each cluster's certificate only
covers its own name, so `auto` uses the first origin whose unauthenticated
`GET /_liveness` returns 200 over verified TLS. Off the VLAN the names do not
resolve. Only these origins ever receive a token, and only after they answered
the probe. Plain HTTP is never accepted for the cache or upstream, not even to
`127.0.0.1`. Only the test suite enables loopback HTTP fixtures, through
`CACHEW_TEST_ALLOW_LOOPBACK_HTTP=1`, which `action.yml` clears on every cachew
step. An untrusted or non-HTTPS `cachew-url` (including the in-cluster
`cachew.cachew.svc.cluster.local`) leaves the plain checkout: no session, token,
wrapper or configuration change.

`cachew-prefetched` reports whether cachew supplied a complete checkout object
seed. `cachew-configured` reports whether global caching was enabled for later
Git operations when the action finished.

## Checkout and later dependencies

On an enabled runner the action creates an owned job-global Git configuration
under `RUNNER_TEMP` (selected with `GIT_CONFIG_GLOBAL`, Git 2.32 or newer). It
starts from the caller's global settings and included files, without inherited
cachew rewrites. The original files are never modified. If the session cannot be
created (no Git, Git older than 2.32, unreadable configuration), the action falls
back to the plain checkout and leaves the job environment unchanged.

The cache phase fetches main repository objects once, then each requested
submodule at its exact gitlink SHA, including nested submodules. Cachew serves
commit SHAs and `refs/pull/*` from its mirror and forwards anything missing
upstream. Full-history checkouts seed all branches and tags. Sparse, LFS, SSH
and non-empty targets skip object seeding.

With `filter`, the main repository is seeded in two cache fetches: the complete
tip commit, then the filtered history at the requested depth. The worktree
therefore needs no lazy fetches. The seed records `origin` as the promisor
remote with the same filter, as standard checkout does, so objects missing
later are fetched lazily from upstream. Submodules are seeded unfiltered, as
standard checkout fetches them.

The pinned official checkout then runs against upstream using the cached objects,
with its own throwaway global file, so its temporary global settings never leak
into the job. It owns ref resolution, worktree creation, outputs and its
credential persistence and cleanup. Afterward the action adds:

```text
url.<cachew>/git/github.com/.insteadOf = https://github.com/
http.<cachew>/....extraHeader          = the checkout token (cache URLs only)
checkout.workers = 0
checkout.thresholdForParallelism = 50
submodule.fetchJobs = 0
gc.auto = 0
```

Tokens are scoped to cache URLs. Upstream credentials stay exactly where
`actions/checkout` puts them (repository-local), so later steps can use other
credentials for other repositories. With `persist-credentials: false` the token
is used only for that checkout's seeding. Steps may write the global
configuration (`git config --global ...`); cachew only adds and removes its own
keys, so those writes survive cache fallback and later checkouts.

The helper needs Node.js 18 or newer. Composite steps do not get the runner's
bundled Node on `PATH`, so `session/prepare.sh` uses `node` from `PATH`, then the
runner's own `externals/node24` or `node20` (found from `RUNNER_TEMP`, which is
`<runner>/_work/_temp`; tt CI VMs run the runner from `/opt/runner` and have no
system Node), then `/__e/node24` or `/__e/node20`, where the runner mounts the
same externals into job containers. If none runs (for example glibc Node in an
Alpine container), the action falls back to the plain checkout without changing
the job.

There is no post step: composite actions cannot have one, and a separate
JavaScript action would need its own pin. The session directory lives in
`RUNNER_TEMP`, which the runner empties when the job completes (also for failed
and cancelled jobs) and again before the next job starts, together with
`actions/checkout`'s own credentials file in the same directory. Until then later
steps and other actions' post steps keep a working `GIT_CONFIG_GLOBAL`. If the
runner process itself dies, the files remain until that runner's next job starts.

The session directory and its files are private to the runner user because the
configuration holds cache-scoped tokens. A step that runs Git as another user
with this environment must set its own `GIT_CONFIG_GLOBAL`, or the job should use
`cachew: false`.

## Upstream fallback

For later Git reads, a small `git` wrapper on `PATH` sends `clone`, `fetch` and
`ls-remote` through cachew once. Other commands go straight to Git without
starting Python. If the cache attempt fails, the same operation is retried once
upstream. Only failures of the cache endpoint itself (connection, TLS, 3xx/5xx,
a stall longer than `cachew-fetch-timeout`, or the one-hour hard deadline) remove
the job's cache rewrite; answers about one repository or ref (401/403/404, a
missing ref, `ls-remote --exit-code` finding nothing) keep caching on. A failed
cache clone's partial destination is removed before the upstream retry.

Native recursive clones, submodule commands, initialized-submodule fetches,
pushes and pulls use upstream directly. Successful cache checkout still needs
upstream ref negotiation; this action does not provide an offline checkout.

## Verification, pins and adoption

```sh
CHECKOUT_ACTION_JS=/absolute/path/to/checkout/dist/index.js \
  python3 -m unittest discover -s .github/actions/cachew-checkout/tests -v
```

Everything runs from this action's own directory, so the consumers' pin to this
action (`ACTION_SHA` in `.github/scripts/utils/cachew_checkout.py`) is the only
pin. To release a change:

1. Commit the action change. In a second commit, set `ACTION_SHA` to the first
   commit and run `python3 .github/scripts/utils/cachew_checkout.py --write`.
   Push both together; `--verify-pins` checks that the pin matches this tree.
2. After merging, re-pin `ACTION_SHA` to the merge commit on `main` and run
   `--write` again (`--verify-pins --require-main`). A commit that is only on a
   deleted branch can be garbage-collected, which would break every consumer.

A checkout with a trailing `# cachew: keep` comment is never migrated (the test
workflow's bootstrap checkout uses this to test the working tree's action).

`--check` and `--write` migrate only checkouts of the version this action
forwards to (v6.0.2). gh-aw generated workflows, intentionally floating
references and the vendored tt-llk tree keep `actions/checkout`.
