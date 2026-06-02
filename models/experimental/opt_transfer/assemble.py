from typing import Callable


def assemble_model(block_runners: list) -> Callable:
    """Compose per-block runners into a single forward."""

    def model(x):
        for run in block_runners:
            x = run(x)
        return x

    return model


def assemble_layers(layer_runners):
    """Stack N decoder-layer runners into one forward. Each runner(prev)->next; unmatched
    ops keep their naive fallback. Generalizes assemble_model to multi-layer models."""

    def model(x):
        for run in layer_runners:
            x = run(x)
        return x

    return model
