"""
MoLE: Mixture-of-Linear-Experts for TT-Metal

A meta-architecture for long-term time series forecasting using TT-NN APIs.

Example:
    >>> from mole_ttnn import MoLE, MoLEConfig
    >>> config = MoLEConfig(num_experts=4)
    >>> model = config.create_model()
    >>> output = model(input_data)
"""

__version__ = "1.0.0"
__author__ = "MoLE Contributors"

# Import main components
from models import (
    MoLE,
    MoLETTNN,
    MoLEConfig,
    DLinear,
    DLinearTTNN,
    Router,
    RouterTTNN,
)

from utils import (
    get_dataloader,
    load_data,
    metric,
    MetricsTracker,
    Trainer,
    TTNNTrainer,
)

__all__ = [
    # Models
    'MoLE',
    'MoLETTNN',
    'MoLEConfig',
    'DLinear',
    'DLinearTTNN',
    'Router',
    'RouterTTNN',
    # Utils
    'get_dataloader',
    'load_data',
    'metric',
    'MetricsTracker',
    'Trainer',
    'TTNNTrainer',
]

def get_version():
    """Return package version"""
    return __version__
