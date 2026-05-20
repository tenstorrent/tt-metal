from .config import AceConfigTTNN, AttentionImplTTNN

# Legacy exports (deprecated — see modules.py module docstring).
from .modules import (  # noqa: F401
    AdaLNZeroTTNN,
    GEGLUMLPTTNN,
    MultiHeadSelfAttentionSDPATTNN,
    MultiHeadSelfAttentionTTNN,
    TransformerBlockTTNN,
)

__all__ = [
    "AceConfigTTNN",
    "AttentionImplTTNN",
    # Deprecated — use dit_decoder_core for inference.
    "AdaLNZeroTTNN",
    "GEGLUMLPTTNN",
    "MultiHeadSelfAttentionTTNN",
    "MultiHeadSelfAttentionSDPATTNN",
    "TransformerBlockTTNN",
]
