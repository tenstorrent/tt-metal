# Local task runtime repair plan

Goal: make the stock local Harbor Docker/Compose task runtime available without changing the harness, task sandbox/network policy, or using credentials.

Confirmation: accessible Docker daemon and Compose support, followed by stock environment prerequisites. Coordinator device-hour box: 0.1 h; no device jobs authorized or needed; expected device use is zero.

Wait for setup/AUTODEBUG.md and verify its causal chain against current host observations before attempting a repair. Check narrowly for existing Docker/Podman/rootless executables, standard local socket locations and daemon processes. Test user namespace support and existing loopback Docker endpoints. Do not scan remote hosts or install a runtime whose required kernel privileges are unavailable.

Hypotheses, in priority order:
1. An existing local daemon/socket can be used without changing host/container configuration.
2. A present rootless runtime can operate with available user namespaces.
3. This execution container lacks the host runtime mount and namespace privileges required by the stock task runtime; host relaunch is necessary.

Record exact checks, outcomes, minimal remediation and remaining uncertainty in setup/AUTOFIX.md. Preserve root AUTOFIX.md and AUTODEBUG.md. No hardware, CI dispatch, credential access or further agents.
