import ast
import re


_DEVICE_SERIES_MESH_MAP = {
    ("tt-galaxy-wh", 32): [4, 8],
    ("tt_galaxy_wh", 32): [4, 8],
    ("n300", 1): [1, 2],
    ("p150b", 1): [1, 1],
}


def _normalize_device_series(device_series):
    if device_series is None:
        return None
    return str(device_series).strip().lower()


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_int_sequence(value):
    """Parse a shape-like value into a list of positive ints, or None if empty/invalid."""
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        parsed = [_safe_int(v) for v in value]
        if any(v is None for v in parsed):
            return None
        parsed = [v for v in parsed if v is not None]
        return parsed or None

    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "()", "None", "null"}:
            return None
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return parse_int_sequence(parsed)
        except (SyntaxError, ValueError):
            pass
        numbers = [_safe_int(num) for num in re.findall(r"-?\d+", text)]
        numbers = [n for n in numbers if n is not None]
        return numbers or None

    return None


def infer_mesh_shape(
    mesh_shape=None, distribution_shape=None, device_ids=None, device_count=None, device_series=None, card_count=None
):
    """Infer a 2D mesh shape from explicit metadata, topology hints, or hardware defaults."""
    parsed_mesh_shape = parse_int_sequence(mesh_shape)
    if parsed_mesh_shape and len(parsed_mesh_shape) == 2 and all(dim > 0 for dim in parsed_mesh_shape):
        return parsed_mesh_shape

    parsed_distribution = parse_int_sequence(distribution_shape)
    if parsed_distribution:
        if len(parsed_distribution) == 2 and all(dim > 0 for dim in parsed_distribution):
            return parsed_distribution
        if len(parsed_distribution) == 1 and parsed_distribution[0] > 0:
            normalized_series = _normalize_device_series(device_series)
            inferred = _DEVICE_SERIES_MESH_MAP.get((normalized_series, _safe_int(card_count)))
            if inferred and inferred[0] * inferred[1] == parsed_distribution[0]:
                return inferred
            return [1, parsed_distribution[0]]

    normalized_series = _normalize_device_series(device_series)
    inferred = _DEVICE_SERIES_MESH_MAP.get((normalized_series, _safe_int(card_count)))
    if inferred:
        return inferred

    count = infer_device_count(
        device_ids=device_ids,
        device_count=device_count,
        mesh_shape=parsed_mesh_shape,
        distribution_shape=parsed_distribution,
    )
    if count == 1:
        return [1, 1]
    if count and count > 1:
        return [1, count]
    return None


def infer_device_count(device_ids=None, device_count=None, mesh_shape=None, distribution_shape=None):
    """Infer device count from explicit ids/count first, then mesh/topology shape."""
    parsed_device_ids = parse_int_sequence(device_ids)
    if parsed_device_ids:
        return len(parsed_device_ids)

    parsed_device_count = _safe_int(device_count)
    if parsed_device_count and parsed_device_count > 0:
        return parsed_device_count

    parsed_mesh_shape = parse_int_sequence(mesh_shape)
    if parsed_mesh_shape:
        product = 1
        for dim in parsed_mesh_shape:
            product *= dim
        if product > 0:
            return product

    parsed_distribution = parse_int_sequence(distribution_shape)
    if parsed_distribution:
        product = 1
        for dim in parsed_distribution:
            product *= dim
        if product > 0:
            return product

    return None


def iter_tensor_placements(value):
    """Yield tensor_placement dicts from arbitrarily nested traced argument structures."""
    if isinstance(value, list):
        for item in value:
            yield from iter_tensor_placements(item)
        return

    if not isinstance(value, dict):
        return

    placement = value.get("tensor_placement")
    if isinstance(placement, dict):
        yield placement

    for nested_value in value.values():
        yield from iter_tensor_placements(nested_value)


def normalize_machine_info(machine_info, arguments=None):
    """Normalize mesh metadata and infer missing shape/count from traced topology hints."""
    normalized = dict(machine_info or {})

    first_placement = None
    if isinstance(arguments, dict):
        for arg_value in arguments.values():
            first_placement = next(iter_tensor_placements(arg_value), None)
            if first_placement:
                break

    distribution_shape = first_placement.get("distribution_shape") if first_placement else None
    placement_mesh_shape = first_placement.get("mesh_device_shape") if first_placement else None

    mesh_shape = infer_mesh_shape(
        mesh_shape=normalized.get("mesh_device_shape"),
        distribution_shape=placement_mesh_shape or distribution_shape,
        device_ids=normalized.get("device_ids"),
        device_count=normalized.get("device_count"),
        device_series=normalized.get("device_series"),
        card_count=normalized.get("card_count"),
    )
    if mesh_shape:
        normalized["mesh_device_shape"] = mesh_shape
    else:
        normalized.pop("mesh_device_shape", None)

    device_count = infer_device_count(
        device_ids=normalized.get("device_ids"),
        device_count=normalized.get("device_count"),
        mesh_shape=mesh_shape,
        distribution_shape=distribution_shape,
    )
    if device_count:
        normalized["device_count"] = device_count
    else:
        normalized.pop("device_count", None)

    if not parse_int_sequence(normalized.get("device_ids")):
        normalized.pop("device_ids", None)

    return normalized
