# AutoDebug: Stage 12 local stock agentic prerequisites

Date: 2026-11-06. Inspection only; no source changes, package installation, daemon startup, privilege escalation, device commands, dispatch, or credential inspection. The root AUTODEBUG.md is preserved.

## Finding: the outer container does not expose a Docker execution backend

**High confidence environment blocker.** Installing the Docker CLI alone does not repair this environment. The stock local agentic path requires a working container backend as well. The smallest practical repair is host-provided Docker access plus CLI/Compose tooling, or running the eval client on a suitably provisioned host. A nested rootless daemon cannot bootstrap with the currently observed namespace policy.

Source paths below are relative to `/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12`, except the outer launcher `/home/mvasiljevic/gemma4-31b-pipeline-v3/container.sh`.

### Direct observations

- `id`: uid/gid 6002 (`bringup`), only group 6002.
- `command -v`: no docker, dockerd, rootlesskit, slirp4netns, fuse-overlayfs, newuidmap or newgidmap on PATH. `unshare` and `sudo` exist; their presence does not establish permission to repair the host.
- `/var/run/docker.sock`, `/run/user/6002`, and `/dev/fuse` are absent. DOCKER_HOST is unset. No Docker credential/config files were read; no exhaustive daemon endpoint discovery was attempted.
- `/proc/self/status`: effective/permitted/inheritable/ambient capabilities are zero; bounding mask `00000000a80425fb` excludes CAP_SYS_ADMIN and CAP_NET_ADMIN. Seccomp mode is 2 with one filter; NoNewPrivs is 0. The filter's actual rules were not inspected.
- `kernel.unprivileged_userns_clone=0`; `user.max_user_namespaces=1019103`. Both subuid and subgid already contain `bringup:100000:65536`.
- `/proc/sys` is mounted read-only; `/sys/fs/cgroup` is cgroup v2 mounted read-only. The user cannot write the namespace sysctl. A container marker exists; UID/GID mappings span the initial numeric range.
- Outer launcher lines 12–27 supplies host network/IPC, shared memory and bind mounts, but no Docker socket, daemon endpoint, supplementary Docker group, privilege flag, or added capabilities. Lines 28–29 optionally expose TT devices; this is unrelated to Docker daemon access. Line 38 invokes Docker **outside** the worker to create it: it does not provide Docker access inside.

### Causal chain in the inspected harness

1. `reference_config/evals/eval_config.py:25–34` defaults Harbor's environment to `docker`; Kubernetes is an explicit alternative.
2. `launchers/run_agentic.py:67–74` selects the agentic Python environment. `workflows/workflow_venvs.py:319–350` installs Harbor editable; it does not provision a Docker engine. `requirements/evals-agentic.txt:8–17` explicitly places agent installation and SWE grading inside task containers.
3. `llm_module/drivers/agentic.py:145–166` builds the Harbor config and invokes it; `:188–193` selects the Harbor driver. `llm_module/agentic/harbor.py:499–533` launches `harbor run`, passing the environment in configuration or `--env`.
4. `harbor.py:209–242` itself executes Docker info/network inspection for task-side endpoint discovery. Failure falls back to `172.17.0.1`; that fallback supplies an address, **not a daemon**. Optional pre-pulling at `:435–495` also invokes Docker, but warning-only handling cannot make task execution work without the backend.
5. Corroboration only: cached Harbor source at `/home/bringup/.cache/uv/git-v0/checkouts/2ac93d8bd318fd02/1da0bfd8/src/harbor/environments/docker/docker.py:174–190` explicitly rejects a missing Docker binary and then checks daemon availability with `info`. This cache was not proven to be the interpreter used by the eventual run; exact exception text is therefore not a revision-matched prediction.

This explains failure before agent/model evaluation. It is not evidence of a model correctness problem. It also does not show that Docker is absent from the physical host: the outer launcher expects host Docker.

## Rootless feasibility

Rootless Docker uses user namespaces and requires mapping helpers plus subordinate ID ranges. The ranges are present; the helpers are absent from PATH. More decisively, unprivileged namespace creation is disabled, and the process cannot change the read-only sysctl. Docker documents this setting as a cause of RootlessKit startup failure. [Rootless prerequisites](https://docs.docker.com/engine/security/rootless/), [troubleshooting](https://docs.docker.com/engine/security/rootless/troubleshoot/).

Installing user-space binaries cannot change that kernel policy. No namespace creation probe was necessary or attempted. A host administrator could change policy and recreate/provision a suitable worker, but this is not an in-container package-only repair. Rootful nested Docker is also not a supported repair with the observed identity and capability bounds.

Do not overstate secondary observations: missing `/dev/fuse` is not independently fatal because other storage drivers exist; lack of a user runtime directory can be repaired; absence of systemd would not alone prohibit manual startup. Read-only cgroups raise resource-enforcement concerns after startup, and seccomp may impose additional restrictions, but neither was proven to be the first failure. Rootless storage, resource limits and networking would still need validation after host-side repair. [Docker rootless limitations](https://docs.docker.com/engine/security/rootless/troubleshoot/).

## Minimal repair boundary and follow-up checks

No repair was executed in this investigation.

1. On the host, expose an authorized daemon socket to the worker (normally `/var/run/docker.sock`) and give uid 6002 access using the socket's actual group/ACL. A launcher recreation is the straightforward boundary; changing its file alone does not update the running container. Provide Docker CLI and the Compose plugin in the worker's configured PATH/plugin search locations. A verified existing remote daemon endpoint is another option, but none was supplied or inspected. Daemon access is host-level authority and should be deliberately provisioned.
2. Preserve daemon-visible paths for any task bind mounts and result directories. The launcher maps REPO and SETUP to identical host paths, making those suitable locations. It maps `SETUP/home` to `/home/bringup`, however: a bind source under `/home/bringup` is not automatically the same host directory. Inspect generated task mounts before using sibling containers; copied/uploaded files do not have the same constraint. A remote daemon adds this same filesystem issue.
3. Verify inference connectivity from task containers, not just from the worker. `harbor.py:245–270` rewrites loopback for in-container agents; terminus-2 runs on the client side per `:55–62`. Host networking of the worker does not make task-container loopback the inference host. Use a reachable endpoint or `AGENTIC_DOCKER_HOST_ADDRESS` when needed, and ensure the server listens on a reachable interface. A daemon on another host especially cannot use its own bridge gateway to reach this server automatically.
4. In a subsequent authorized execution phase, check CLI/Compose versions, daemon access as uid 6002, task mounts, image acquisition, and a small stock task's setup/verifier completion before a full eval. Check daemon disk/CPU/RAM and network capacity for the configured concurrency. These checks were not run here.

Thus: **not repairable solely from this unprivileged container with its current mounts/policy.** Host-side socket provisioning or an externally provisioned execution backend is required. This statement does not rule out an administrator arranging access while the worker remains running; it rules out treating CLI installation or rootless bootstrap as a complete local fix.

## Revision and coverage limits / other discrepancies

- Inspected checkout HEAD is `7a51c2f20f51e7508d8ea365d09092d00bbd5fcb`; tracked worktree status was clean. Inputs pin release spec revision `411057222c2dc5a4268e693b0bd57bc0e2571f38`, which is not a tree object available in this checkout. No fetch was performed. Do not present the inspected registry's limits as independently verified limits of that pinned release.
- The inspected Gemma registry (`reference_config/evals/eval_config.py:11935` onward) includes Terminal-Bench 2 with terminus-2 and a 24-trial configuration, plus common/vision evals. Docker findings apply to the containerized agentic path, not automatically to every accuracy suite. No old Gemma implementation was read.
- `workflows/workflow_venvs.py:45–46` names Harbor's `tt-inference-server` branch. Record the actually installed Harbor commit before claiming reproducibility; the inspected cache is supporting evidence only.
- Kubernetes is a supported configurable alternative, but its infrastructure/access were not inspected and it is not a local Docker repair. No credentials were sought.

## Adjudication after first draft

Rechecked each headline against launcher and harness: no socket forwarding exists; Python dependency provisioning supplies no engine; endpoint fallback cannot supply one; rootless namespace policy is disabled even though subordinate IDs exist. Demoted missing FUSE, cgroup delegation, exact cached Harbor exception behavior, and remote-path/network issues to conditional follow-ups. No hardware-dependent claim is made. The remaining uncertainty is provisioning outside this container and exact release/installed-harness identity, not the observed absence of local Docker prerequisites.
