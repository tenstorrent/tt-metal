# AutoDebug: missing effective-configuration endpoint

The first timed launch reached health readiness with the full model loaded, then failed with HTTP 404 for `/server_info?config_format=json`. The hook correctly stopped its owned server. This is an HTTP route-registration failure, not evidence of invalid weights, precision, context, or hardware.

The actual installed core is `/home/container_app_user/tt-metal/python_env/lib/python3.10/site-packages/vllm` (0.26.0+empty), not the nested checkout. Its `entrypoints/openai/api_server.py:237-240` invokes `register_vllm_dev_api_routers` only when `envs.VLLM_SERVER_DEV_MODE` is true. Its `entrypoints/serve/__init__.py` then attaches `dev.server_info.api_router`. The installed `envs.py` defaults this variable to `0`. Initial source inspection used the nested checkout; the hypothesis is confirmed against the installed wheel's actual registration gate and test below executes that installed source. The benchmark launcher did not set it. The always-registered health router explains why readiness succeeded while configuration lookup returned 404.

The installed `entrypoints/serve/dev/server_info/api_router.py` obtains `raw_request.app.state.vllm_config` and serializes it with the `VllmConfig` Pydantic adapter when `config_format=json`. This is the actual effective engine configuration required by the existing validator; it should not be replaced by requested argv.

Hypothesis check: execute the actual route-registration function using fake router modules/app, toggling only `VLLM_SERVER_DEV_MODE`; server_info is absent when false and present when true. No vLLM engine or device import is needed. Preserve this check with profile tests.

Smallest repair: set `VLLM_SERVER_DEV_MODE=1` in both benchmark profile environments, retain that value in the launch snapshot, and bind this stage-owned HTTP server to localhost. No model, precision, cache, context, trace, scheduler or inference setting changes. This enables vLLM's configuration-reporting route (and other development endpoints); none of those other endpoints is invoked. Live route availability must still be verified on the next timed launch.

## AutoFix validation

Four host tests pass after the repair (`configuration_route_after.log`). The added test executes the installed wheel's actual API gate and development-router registration AST against mock router modules; it verifies server_info absent/attached with the flag false/true and that both profile plans set the flag and localhost bind. `configuration_route_before.log` preserves the failing plan check.

The configuration HTTP timeout is 30 seconds because the route also collects host environment information. Launch evidence now retains `VLLM_SERVER_DEV_MODE`. Server identity distinguishes the imported core wheel (0.26.0+empty, commit unknown/null) from the actual plugin checkout commit, and records installed API-server/benchmark source hashes. The metadata probe reads source and distribution metadata without importing the platform plugin or opening devices. No hardware was used in this diagnosis or validation.
