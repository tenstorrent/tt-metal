# Cachew checkout

Replace an `actions/checkout` step with this composite action. Pin the tt-metal
commit containing the action, then keep the existing checkout inputs:

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

An initial checkout needs the remote action reference above. A local reference
such as `./.github/actions/cachew-checkout` works only after a bootstrap checkout.
The test workflow uses a bootstrap checkout and a separate target directory.

All 20 inputs and the `ref` and `commit` outputs from
[`actions/checkout` v6.0.2](https://github.com/actions/checkout/tree/de0fac2e4500dabe0009e67214ff5f5447ce83dd)
are forwarded. `cachew-prefetched` reports whether cachew supplied a complete
object seed. It does not distinguish a cache hit from a pull-through cache miss.

| Additional input | Default | Meaning |
| --- | --- | --- |
| `cachew-url` | `https://cachew.ci.tenstorrent.net` | HTTPS origin nearest the runner |
| `cachew-health-timeout` | `2` | Seconds allowed for the single liveness request, maximum 30 |
| `cachew-fetch-timeout` | `120` | Seconds allowed per cache fetch, maximum 600 |

Use `https://cachew.dev.tenstorrent.net` in dev,
`https://cachew.ci.tenstorrent.net` in Austin production, and
`https://cachew.yyz2.ci.tenstorrent.net` in Toronto production. Public HTTPS
certificates are checked normally. HTTP is accepted only for loopback test servers.
Only these three cache origins are allowed to receive the checkout token. Loopback
HTTP fixtures require both the cache and upstream to be local.

## Fetch and fallback

The action makes one unauthenticated `GET /_liveness` request. Cachew registers
that endpoint in `cmd/cachewd/main.go` on its `tt-main` branch. Only HTTP 200
permits a cache attempt. An error, timeout, or redirect goes straight to upstream.

The cache phase fetches the requested repository once, then fetches each requested
submodule at its gitlink SHA. Recursive traversal includes nested submodules and
relative GitHub URLs. Every fetch has a deadline. The first failure discards the
staged seed and immediately continues with upstream checkout. There are no cache
retries, retry sleeps, or recursive Git clone/update commands in this phase.

The pinned official checkout action always runs against canonical upstream URLs.
It reuses the prefetched objects and owns final ref resolution, working-tree
creation, checkout outputs, credential persistence, and post-job cleanup. This
still needs upstream ref negotiation after a successful cache attempt. Cachew
reduces Git object transfer; it does not provide an offline GitHub checkout.

Cache fetch authentication exists only in the Git subprocess environment. No
cache URL rewrite or token is written to repository or global Git config. The
health probe receives no token, and cache requests cannot redirect credentials.

Existing nonempty targets, sparse or filtered checkouts, LFS, SSH keys, and Windows
runners use upstream directly. Missing Python 3.10+, Git, curl, or any prefetch runtime error
also permits upstream checkout. Unsupported submodule hosts or unsafe paths stop
the cache attempt. These cases retain the official checkout behavior.

This follows the [dev-cluster example](https://github.com/230470239cf4v7r09237m/ci-test/blob/dfffe07b05d22d08270f44eee97b84a8e435cbd0/.github/workflows/run-tt-metal-tests.yaml)
for cachew Git routes and authentication. The cache transport is confined to the
prefetch process because both standard checkout and native recursive submodule
update can retry failed fetches.

## Verification

The focused CI workflow runs real smart-HTTP Git fixtures and the exact pinned
checkout bundle, including its post handler. Tests cover nested submodules whose
names differ from their paths, unchanged object files after checkout, blocked
upstream submodule transport, health and fetch failures, timeout and disconnect
request counts, tags and full history, and credential cleanup.

To run locally, obtain `dist/index.js` from the pinned checkout commit and run:

```sh
CHECKOUT_ACTION_JS=/absolute/path/to/checkout/dist/index.js \
  python3 -m unittest discover -s .github/actions/cachew-checkout/tests -v
```

The test workflow also offers an optional manual live smoke checkout on a runner
with access to the selected cachew endpoint. Production checkout callers are
unchanged by this action's introduction.
