from typing import Callable


def assemble_layers(layer_runners) -> Callable:
    """Stack N decoder-layer runners into one forward. Each runner(prev)->next; unmatched
    ops keep their naive fallback. Generalizes assemble_model to multi-layer models."""

    def model(x):
        for run in layer_runners:
            x = run(x)
        return x

    return model


# assemble_model is a thin alias kept for backwards compatibility.
assemble_model = assemble_layers
