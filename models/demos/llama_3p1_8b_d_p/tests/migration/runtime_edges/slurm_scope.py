EXPECTED_STATE = "RUNNING"


def _lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def expand_nodes(node_expression, runner):
    nodes = _lines(runner(["scontrol", "show", "hostnames", node_expression]))
    if not nodes:
        raise ValueError(f"empty node expansion for {node_expression!r}")
    return nodes


def validate_allocation(record, target_node, runner, *, expected_owner):
    lines = _lines(record)
    if len(lines) != 1:
        raise ValueError(f"expected one allocation record, got {lines!r}")
    fields = lines[0].split("|", 2)
    if len(fields) != 3:
        raise ValueError(f"malformed allocation record {lines[0]!r}")
    owner, state, node_expression = fields
    if owner != expected_owner:
        raise ValueError(f"unexpected owner {owner!r}")
    if state != EXPECTED_STATE:
        raise ValueError(f"unexpected state {state!r}")
    nodes = expand_nodes(node_expression, runner)
    if target_node not in nodes:
        raise ValueError(f"target {target_node!r} is absent from allocation {nodes!r}")
    return {
        "owner": owner,
        "state": state,
        "node_expression": node_expression,
        "expanded_nodes": nodes,
        "target_node": target_node,
    }


def target_step_lines(record, target_node, runner):
    selected = []
    for line in _lines(record):
        fields = line.split("|", 2)
        if len(fields) != 3:
            raise ValueError(f"malformed step record {line!r}")
        if target_node in expand_nodes(fields[2], runner):
            selected.append(line)
    return selected


def singleton_srun(job, target_node, payload):
    return [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--exact",
        "--cpus-per-task=1",
        "--jobid",
        job,
        "-w",
        target_node,
        *payload,
    ]
