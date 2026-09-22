# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Portable setup/ownership for the unchanged source96 backend.

Call configure_tracking before importing TTNN (including upstream conftests).
Caller-owned devices must already reserve DEVICE_OPTIONS and have program cache
enabled for their whole lifetime. This module never configures/closes a borrowed
device. Retained owners indicate unresolved native state: do not reset, reopen,
close, or claim recovery. Inspect retained_owners() for the owning model/device.
"""

import os
import sys

_RETAINED = []


def configure_tracking():
    required = {"TT_METAL_TRACE_ALLOC_TRACKING": "1", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0"}
    if "ttnn" in sys.modules and any(os.environ.get(k) != v for k, v in required.items()):
        raise RuntimeError("Set trace allocation tracking before importing TTNN")
    os.environ.update(required)


def retained_owners():
    return tuple(_RETAINED)


def _note(primary, message):
    try:
        primary.add_note(message)
    except BaseException:
        pass


class RuntimeOwner:
    """One device lifetime; bind each model before issuing any request."""

    def __init__(self, device=None):
        self.device = device
        self.owned = device is None
        self.models = []
        self.closed = False
        self.cleanup_errors = []
        self._check_admission()

    def _check_admission(self):
        if self.closed or self in _RETAINED:
            raise RuntimeError("Owner is closed or retained")
        if self.device is not None and any(o.device is self.device for o in _RETAINED):
            raise RuntimeError("Device has an unresolved retained owner")

    def bind(self, model):
        self._check_admission()
        if not any(model is existing for existing in self.models):
            self.models.append(model)
        return model

    def open(self, device_id):
        if self.closed or self in _RETAINED or not self.owned or self.device is not None:
            raise RuntimeError("Owner already has a device, is closed, or is retained")
        configure_tracking()
        import ttnn

        if __package__:
            from .backend import DEVICE_OPTIONS
        else:
            from backend import DEVICE_OPTIONS
        if not self.owned or self.device is not None:
            raise RuntimeError("Owner already has a device")
        self.device = ttnn.open_device(device_id=device_id, **DEVICE_OPTIONS)
        try:
            self.device.enable_program_cache()
        except BaseException as primary:
            self.finish(primary)
            raise
        return self.device

    def _retain(self):
        if self not in _RETAINED:
            _RETAINED.append(self)

    def finish(self, primary=None):
        if self.closed:
            return
        errors = []
        unresolved = any(o.device is self.device for o in _RETAINED)
        # Native state is authoritative even if no model was registered. An
        # observation failure cannot establish that closing the device is safe.
        if self.device is not None:
            try:
                import ttnn

                unresolved = bool(ttnn.is_trace_capture_active(self.device)) or unresolved
            except BaseException as error:
                errors.append(error)
                unresolved = True
        for model in self.models:
            try:
                # generate() clears _decode_trace in its finally block; failed
                # source96 owners remain reachable through _trace_failures.
                if getattr(model, "_trace_failures", ()):
                    unresolved = True
                    continue
                traces = list(getattr(model, "_last_warmup_owners", ()))
                current = getattr(model, "_decode_trace", None)
                if current is not None and not any(current is t for t in traces):
                    traces.append(current)
                for trace in traces:
                    if getattr(trace, "unresolved", False):
                        unresolved = True
                    elif not unresolved:
                        trace.close()
                        if getattr(trace, "unresolved", False) or getattr(trace, "trace_id", None) is not None:
                            unresolved = True
                unresolved = bool(getattr(model, "_trace_failures", ())) or unresolved
            except BaseException as error:
                errors.append(error)
                unresolved = True
        if not unresolved and not errors and self.device is not None:
            try:
                # Recheck immediately before close, after model cleanup.
                unresolved = bool(ttnn.is_trace_capture_active(self.device))
            except BaseException as error:
                errors.append(error)
                unresolved = True
        if unresolved or errors:
            self._retain()
            errors.append(RuntimeError("Native trace cleanup unresolved; owner/device retained"))
        elif self.owned and self.device is not None:
            try:
                ttnn.close_device(self.device)
                self.closed = True
            except BaseException as error:
                self._retain()
                errors.append(error)
        else:
            self.closed = True
        self.cleanup_errors.extend(errors)
        if errors:
            if primary is not None:
                for error in errors:
                    _note(primary, "Runtime cleanup: " + type(error).__name__)
            else:
                raise errors[0]

    def __enter__(self):
        self._check_admission()
        return self

    def __exit__(self, kind, error, traceback):
        self.finish(error)
        return False
