import itertools
import pytest
from loguru import logger

OP_MARKER_PREFIX = "op_"
OPS = ["reshape", "linear"]

MODEL_MARKER_PREFIX = "model_"
MODELS = ["deepseek_v3", "llama"]


def pytest_configure(config):
    # Add op and model markers
    existing_markers = config.getini("markers")

    for marker_name, marker_type in itertools.chain(
        *(
            [(f"{marker_prefix}{marker_item}", marker_type) for marker_item in marker_items]
            for marker_prefix, marker_items, marker_type in [
                (OP_MARKER_PREFIX, OPS, "op"),
                (MODEL_MARKER_PREFIX, MODELS, "model"),
            ]
        )
    ):
        print(marker_name, marker_type)
        assert not any(
            marker_line.startswith(f"{marker_name}:") or marker_line.startswith(f"{marker_name}(")
            for marker_line in existing_markers
        ), f"Marker {marker_name} for {marker_type}s is already taken"
        config.addinivalue_line("markers", f"{marker_name}: mark test for the {marker_name} {marker_type}")

    print(config.getini("markers"))


def pytest_itemcollected(item):
    # Check if a test has a model and an op marker
    exiting_due_to_improper_marking = False
    for marker_prefix, marker_type, marker_collection in (OP_MARKER_PREFIX, "op", OPS), (
        MODEL_MARKER_PREFIX,
        "model",
        MODELS,
    ):
        if len([marker for marker in item.own_markers if marker.name.startswith(marker_prefix)]) != 1:
            logger.error(
                f"A test should specify exactly one {marker_type} marker (one of {[f'{marker_prefix}{item}' for item in marker_collection]})"
            )
            exiting_due_to_improper_marking = True
    if exiting_due_to_improper_marking:
        pytest.exit(
            "Improper test markings",
            returncode=1,
        )
