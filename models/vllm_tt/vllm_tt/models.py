import logging

logger = logging.getLogger(__name__)

_registered = False


def register_tt_models():
    global _registered
    if _registered:
        return
    _registered = True

    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "TTPluginNoOpModel",
        "vllm_tt.noop_model:NoOpModel",
    )
    logger.info("Registered TT plugin models: TTPluginNoOpModel")


def register_tt_models_plugin():
    """Entry point for vllm.general_plugins group."""
    register_tt_models()
