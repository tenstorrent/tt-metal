# AutoFix: local Harbor runtime unavailable

Date: 2026-11-06. Effort: `runs/local-runtime-repair`. Device hours used: **0** (box 0.1 h; no device jobs). Status: **done; no change; problem open; confirmation failed**.

## Contract and diagnosis

Stock local Harbor must launch its configured Docker task environments with task-defined CPU/memory and unchanged sandbox/network policy. `setup/AUTODEBUG.md` diagnoses absent Docker backend exposure and unavailable user namespaces. This was checked against the current outer `container.sh` and `tt-inference-server-stage12/llm_module/agentic/harbor.py`. The launcher has no Docker socket mount; Harbor uses Docker for task environments, not merely an optional model endpoint probe.

## Hypotheses and checks

1. **An existing local daemon is usable:** no docker, podman, dockerd or rootlesskit executable on PATH; the four standard Docker/Podman sockets are absent; DOCKER_HOST is unset; no daemon process was seen; loopback ports 2375 and 2376 both refuse connections. No remote host was scanned.
2. **Rootless bootstrap can supply the missing daemon:** `unshare -Ur true` fails with EPERM. `kernel.unprivileged_userns_clone` is 0. Mapping/helper executables are absent. Installing a CLI or rootless userspace cannot repair this observed namespace denial.
3. **Rootful nested daemon can be enabled with existing privilege:** `sudo -n unshare -m true` and `sudo -n unshare -n true` both fail with EPERM. Rootful namespace creation is unavailable too. No daemon installation or startup was attempted.

Exact checks and results are in `runs/local-runtime-repair/result.json`; the ineffective multi-target findmnt invocation is explicitly recorded and not used as evidence. These checks corroborate the diagnosis. There is no source-level fix for the earliest failing prerequisite inside the allowed boundary.

## Minimal host-side remediation

The host operator must provision an authorized Docker backend to this worker, normally by recreating it with `--volume /var/run/docker.sock:/var/run/docker.sock` and access for uid6002 using the host socket's actual group (for example `--group-add <socket-gid>`) or ACL. Provide the Docker CLI and Compose plugin in the worker. This grants daemon authority and must be supplied deliberately by the host; changing a launcher file does not alter the running worker. No launcher changes were made.

Preserve the checkout, Codex home, logs and run directories across recreation. Keep task bind-source paths identical and visible to the host daemon; the existing REPO/SETUP identity mounts satisfy that for their contents, whereas `/home/bringup` is a remapped path and requires checking. Keep stock task sandbox/network policies and task CPU/memory values unchanged.

After provisioning, as uid6002 run bounded `docker version`, `docker info` and `docker compose version`. Then use the stock harness preflight to verify task image acquisition, daemon-visible mounts/results, task-container connectivity to the served model, task-declared resource limits, disk admission and one stock environment setup/verifier. Docker task loopback is not the worker loopback: use the harness's existing endpoint mapping configuration when required. No broad network policy changes are proposed.

## Outcome and remaining uncertainty

**AutoFix could not restore the local runtime within this environment.** CLI-only installation is not a repair. Host provisioning/recreation is the remaining action; physical host daemon availability and successful task execution afterward remain unverified. This is an execution environment blocker, not evidence against model correctness. No hardware, CI, credential access, harness/source edits, package installs or root report changes occurred.

## Pinned release source clarification

The coordinator separately fetched the pinned `reference_config/evals/eval_config.py` through the GitHub API into `setup/pinned_eval_config.py` and verified GPQA release values of 14400 seconds and 126976 output tokens. This resolves the AutoDebug report's pinned-config inspection gap for those values; it does not supply a local task backend. This effort did not fetch or independently re-audit that configuration.
