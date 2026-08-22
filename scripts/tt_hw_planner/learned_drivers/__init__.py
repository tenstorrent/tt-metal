"""Per-model learned capture drivers (LLM-drafted, persisted by
auto_capture_driver_onboard.py). Each .py file in this directory registers
its driver via the @register_capture_driver decorator at import time.

This package is loaded at the start of the capture phase via
auto_capture_driver_onboard.load_learned_drivers().
"""
