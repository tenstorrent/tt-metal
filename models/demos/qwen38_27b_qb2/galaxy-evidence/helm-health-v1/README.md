# Authenticated Qwen Helm probe correction

The generated Qwen overlay previously inherited the chart's `/v1/models`
liveness path. With `auth.apiKey` set, the pinned vLLM 0.26 authentication
middleware returns 401 for that unauthenticated kubelet request. A healthy
server would therefore fail liveness after startup and be restarted.

TTIS commit `2485b039b` on `anatarajan/qwen38-galaxy-release-20261009`
sets the Qwen overlay's liveness and readiness paths to `/health`; the chart
derives startup from readiness. The installed health handler calls
`engine_client.check_health()` and returns 503 for a dead engine. The fix does
not disable authentication on inference or model-listing routes.

Four render regression cases failed before the correction. All 19 bundle
tests passed afterward, covering both precision policies, authentication on
and off, unchanged API-key injection, immutable image references, full-device
claims and read-only weights. Ruff check/format and git diff checks passed.

[values.yaml](values.yaml) is regenerated from the actual native G0 receipt and
exact model-source pin. [render-audit.json](render-audit.json) records the full
TTIS revision, corrected probes and the unchanged runtime ModelSpec hash.
The G0 receipt, runtime ModelSpec and verifier are byte-identical to the original
build inputs. The checkpoint mount in these values is specific to the allocated
host; SJC3 must supply its own local path and hardware qualification.

[authentication-audit.py](authentication-audit.py) ran against the installed
source on the allocated host without importing vLLM or opening devices. It
extracts the **unmodified** authentication class and guarded-prefix constant
from the AST, uses real Starlette request/response handling, and supplies a
synthetic downstream ASGI application. The source hash and four observed
statuses are in [authentication-audit.json](authentication-audit.json):
unauthenticated `/v1/models` is 401; a valid test token reaches the downstream
application; `/health` bypasses the authentication guard. This is a middleware
reproduction, not a live model endpoint or Kubernetes validation.

The experimental image digest is unchanged and remains unqualified:
`sha256:0b11f045bf089088a62b6e3c1aeb9b64cc72b74a632935f25203609b1e023579`.
Regenerate values using the corrected preparation script. For the original
bundle, `--set defaults.probes.liveness.path=/health` fixes its inherited path.
The image is not yet published to its intended registry.

The attempt to move no-device container checks ahead of the queue was rejected
by the existing disk budget: build host .34 had 28,067,233,792 bytes available
versus 33,510,177,281 bytes required including the 8-GiB reserve. None of its
three existing Metal images shares a layer prefix with the new image, so no
snapshot reuse was credited. No unrelated image or data was deleted; the
persistent .98 container-check queue remains unchanged.
