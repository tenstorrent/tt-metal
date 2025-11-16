"""Reference PyTorch implementation of YOLOS-small"""

from .config import YolosConfig, get_yolos_small_config
from .modeling_yolos import YolosForObjectDetection

__all__ = ["YolosConfig", "get_yolos_small_config", "YolosForObjectDetection"]
