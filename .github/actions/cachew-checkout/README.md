# Cachew checkout

Use this action in place of `actions/checkout`, retaining existing checkout inputs:

```yaml
- name: Checkout
  id: checkout
  uses: tenstorrent/tt-metal/.github/actions/cachew-checkout@<full-tt-metal-commit-sha>
  with:
    cachew-url: https://cachew.dev.tenstorrent.net
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
| `cachew-url` | `https://cachew.ci.tenstorrent.net` | HTTPS origin nearest the runner |
| `cachew-health-timeout` | `2` | Seconds for the single liveness request, maximum 30 |
| `cachew-fetch-timeout` | `120` | Seconds per cache Git operation, maximum 600 |

Use `https://cachew.dev.tenstorrent.net` in dev,
`https://cachew.ci.tenstorrent.net` in Austin production, and
`https://cachew.yyz2.ci.tenstorrent.net` in Toronto production. Only these three
origins receive tokens. Public TLS certificates are verified. HTTP is accepted
only when both upstream and cache are loopback test servers.

`cachew-prefetched` reports whether cachew supplied a complete checkout object
seed. `cachew-configured` reports whether global caching was enabled for later
Git operations when the action finished. Neither distinguishes an existing cache
hit from a pull-through miss. A later cache failure disables caching for the job.

## Checkout and later dependencies

The action first creates an owned job-global Git configuration under
`RUNNER_TEMP`. It preserves the caller's global settings and included files while
excluding inherited cachew rewrites during upstream checkout. The original global
files remain untouched. Checkout's `safe.directory` entries survive activation
and repeated checkouts, including upstream fallback.

One unauthenticated `GET /_liveness` request must return HTTP 200 before using
cachew. The cache phase fetches main repository objects once, then fetches each
requested submodule at its exact gitlink SHA, including nested submodules. Full
history checkouts seed all branches and tags. Submodules retain Git's normal tag
following. A failed request, redirect, timeout, or Git operation stops that cache
attempt immediately, without retry sleeps or a second cache fetch.

The pinned official checkout runs against upstream using the cached objects. It
owns final ref resolution, worktree creation, outputs, and its credential cleanup.
Afterward, the action applies the global configuration used by the
[tested dev-cluster workflow](https://github.com/230470239cf4v7r09237m/ci-test/blob/dfffe07b05d22d08270f44eee97b84a8e435cbd0/.github/workflows/run-tt-metal-tests.yaml):

```text
url.<cachew>/git/github.com/.insteadOf = https://github.com/
checkout.workers = 0
checkout.thresholdForParallelism = 50
submodule.fetchJobs = 0
gc.auto = 0
```

These are real global Git settings, selected through `GIT_CONFIG_GLOBAL` for the
rest of the job. They cover CMake `FetchContent` and other repositories fetched
after checkout. Both upstream and rewritten cache URLs have scoped extra headers.
With `persist-credentials: true`, the primary checkout token supplies the default
authentication for later dependencies. Secondary repository tokens are scoped to
their repositories. A secondary checkout cannot replace another repository's
credentials. With `persist-credentials: false`, that invocation's token is used
only for its checkout attempt and is not retained in global headers.

A registered post action removes the owned configuration, headers, and Git wrapper
after the last checkout invocation's cleanup. Repeated action invocations share a
session; the caller's original global configuration is preserved byte-for-byte.
An include-only proxy in `RUNNER_TEMP` preserves default HOME/XDG settings for
remaining post actions when no explicit global config file was selected. It
contains no copied credentials and is removed with the runner's temporary files.

## Upstream fallback

A failed probe or initial cache fetch leaves checkout and later dependencies on
upstream. Sparse, filtered, LFS, SSH, and existing-target checkouts skip object
prefetch but can still enable global caching for later HTTPS dependencies after a
successful liveness check. Windows runs standard checkout directly. Missing
Python 3.10+ or curl leaves caching disabled.

For later Git reads, a small wrapper on `PATH` runs ordinary `clone`, `fetch`, or
`ls-remote` once through cachew. On an error or deadline, it removes the owned
global rewrite and retries the operation once against upstream. All later Git
operations stay upstream for that job. Upstream authentication and performance
settings remain available. This includes cache failures during CMake configuration.

Git's native recursive clone and submodule commands can retry failed clones
internally. The wrapper sends those operations directly upstream to prevent hidden
cache retries. Initialized-submodule fetches, pushes, and pulls also use upstream.
The checkout action's own requested submodules are cached through its explicit
single-attempt traversal. Successful cache checkout still needs upstream ref
negotiation; this action does not provide an offline GitHub checkout.

## Verification and adoption

The CI workflow exercises real smart-HTTP repositories and the exact pinned
checkout main/post bundle. Tests cover refspecs, tags and full history, nested
submodules, inherited rewrites, token scoping, repeated checkout cleanup, actual
CMake dependency clones, and cache failure during later CMake clone/fetch.

```sh
CHECKOUT_ACTION_JS=/absolute/path/to/checkout/dist/index.js \
  python3 -m unittest discover -s .github/actions/cachew-checkout/tests -v
```

The optional manual live smoke workflow requires both a cache seed and enabled
global caching. Upstream fallback alone cannot make that smoke test pass.

All workflow and composite-action consumers use a remote reference pinned to the
published action commit, including vendored LLK workflows. The wrapper's internal
upstream checkout remains the sole direct `actions/checkout` invocation.

Run `python3 .github/scripts/utils/cachew_checkout.py --check` to verify adoption,
or pass `--write` to migrate consumers. The agentic compiler hook applies the same
migration after compilation, updating generated steps and manifests together.
After a direct `gh aw compile`, run the migration with `--write` before committing.
