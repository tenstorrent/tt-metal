# SWE smoke network validity diagnosis

## Finding

Strong source/runtime evidence supports missing task Python environment activation, rather than treating both installs as unexplained gratuitous model behavior. The trial agent executes the base Miniconda Python environment instead of the SWE-bench testbed environment. Both old trajectories remain quarantined; a repair cannot retroactively make their scores valid.

## Evidence chain

Collected trial roots are under runs/smoke2-swe-native-parser/collected/.../swe_bench_verified_20261106T073358/. Django trial django__django-11265__ohWF75G and Pylint trial pylint-dev__pylint-7080__CJczXtL use mini-swe-agent2.2.8. Their emitted job Harbor config contains agent_kwargs.config.model only; there is no config.environment.env.PATH. Each mini-swe-agent.trajectory.json info.config.environment has only pager/progress variables, no Python environment selection.

Django messages2/3: `pip show django` returns package not found. Messages6/7: model runs `pip install -e .`, obtaining dependencies. Later interpreter stack at message33 is /opt/miniconda3/lib/python3.11, the base environment. Pylint message39: `python3 -m pylint` reaches /testbed/pylint but fails `ModuleNotFoundError: No module named tomlkit`. Messages40/41: `pip install -r requirements_test_min.txt || pip install tomlkit astroid` downloads packages; already-present package locations are /opt/miniconda3/lib/python3.11/site-packages, confirming base rather than /opt/miniconda3/envs/testbed.

Exact task configs pin harbor-datasets86723674f04e4209ac479d0fb75d9d9f44b4377e. Retrieved only environment/Dockerfile and task.toml for these two tasks (copies alongside this report). Dockerfiles use official swebench/sweb.eval.x86_64.django_1776_django-11265:latest and swebench/sweb.eval.x86_64.pylint-dev_1776_pylint-7080:latest, WORKDIR/testbed, uv install, and mkdir/logs. They do not select the task conda environment. CPU1/memory4G/storage10G are task-defined and unchanged.

Local pinned mini-swe2.2.8 source src/minisweagent/environments/local.py LocalEnvironment.execute calls subprocess.run(shell=True, env=os.environ | self.config.env). It does not activate conda; ordinary shell commands therefore inherit base PATH unless config.environment.env supplies it. Harbor e76c762f src/harbor/agents/installed/mini_swe_agent.py writes the supplied config as YAML and passes `-c` to mini; its launcher puts user-local tool executables on PATH but does not activate testbed. TTI llm_module/agentic/harbor.py _apply_mini_swe_agent_defaults sets model timeout/cost/drop_params only. Thus there is no hidden task interpreter activation in the inspected route.

## Minimal supported correction (not applied)

Use existing model-specific SWE agent_kwargs.config.environment.env.PATH to prepend /opt/miniconda3/envs/testbed/bin while retaining ordinary system executables. This is supported by stock mini config and already expressible through the unchanged Harbor configuration route. It neither changes task selection, network policy, task images, requirements, solution code nor task resource allocation. A model-entry correction is sufficient to test the diagnosis; do not alter generic sandbox policy or bake task solutions/dependencies into stage code.

## Remaining uncertainty and confirmation

No Docker/device access was used. The exact live base-image contents and installed testbed dependencies were not inspected, so do not claim that every needed dependency already exists or that PATH alone prevents all future model pip calls. Strong evidence establishes wrong interpreter use; stock SWE image design suggests testbed is intended, but actual interpreter/package availability needs live confirmation. Fastest discriminating check on the granted CI task environment is read-only `command -v python python3 pip`, `python -c 'import sys; print(sys.executable)'`, and task-required package imports under the selected environment. Do not modify task tests or solutions. The next normal original-cohort smoke must still undergo trajectory audit; any network access remains quarantined, including after this correction.

No source changes before coordinator review. This report diagnoses an integration candidate; raw rewards0 are not valid scored failures under the quarantine rule and no nightly advance is supported.
