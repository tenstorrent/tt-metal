from typing import Callable


def assemble_model(block_runners: list) -> Callable:
    """Compose per-block runners into a single forward."""

    def model(x):
        for run in block_runners:
            x = run(x)
        return x

    return model
