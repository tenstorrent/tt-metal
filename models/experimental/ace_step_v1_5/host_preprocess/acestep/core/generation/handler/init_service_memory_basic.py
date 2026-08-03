# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Memory and device-check helpers for initialization/offload flows."""

import ctypes
import gc
import os
import platform

try:
    import resource
except ImportError:
    resource = None

import torch
from loguru import logger

# Cached libc handle for mallopt/malloc_trim calls (Linux only).
_LIBC = None


def _get_libc():
    """Return a cached ctypes handle to libc.so.6 (Linux only)."""
    global _LIBC
    if _LIBC is None and platform.system() == "Linux":
        try:
            _LIBC = ctypes.CDLL("libc.so.6")
        except Exception as exc:
            logger.debug("[memory] Failed to load libc.so.6: {}", exc)
    return _LIBC


def _apply_malloc_mmap_threshold() -> None:
    """Set glibc M_MMAP_THRESHOLD to 128 KB so large freed blocks go back to OS.

    This is a process-wide setting that affects all libraries.  128 KB is
    chosen as a compromise: large enough to avoid excessive mmap/munmap
    syscall overhead for moderate allocations, yet small enough that
    PyTorch tensor storage freed during CPU-offload is returned to the OS
    promptly instead of being retained in the glibc arena.
    """
    if platform.system() != "Linux":
        return
    libc = _get_libc()
    if libc is None:
        return
    try:
        # M_MMAP_THRESHOLD = -3; 128 KB threshold
        _ = libc.mallopt(-3, 131072)
        logger.debug("[memory] Set M_MMAP_THRESHOLD=131072 for immediate OS reclaim of large frees")
    except Exception as exc:
        logger.debug("[memory] mallopt not available: {}", exc)


_apply_malloc_mmap_threshold()


class InitServiceMemoryBasicMixin:
    """Memory cache, sync, and tensor-device utility helpers."""

    def _empty_cache(self):
        """Clear accelerator memory cache (CUDA, XPU, or MPS)."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _synchronize(self):
        """Synchronize accelerator operations (CUDA, XPU, or MPS)."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _memory_allocated(self):
        """Get current accelerator memory usage in bytes, or 0 for unsupported backends."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0

    def _max_memory_allocated(self):
        """Get peak accelerator memory usage in bytes, or 0 for unsupported backends."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
        return 0

    def _is_on_target_device(self, tensor, target_device):
        """Check if tensor is on the target device (handles cuda vs cuda:0 comparison)."""
        if tensor is None:
            return True
        try:
            if isinstance(target_device, torch.device):
                target_type = target_device.type
            else:
                target_type = torch.device(str(target_device)).type
        except Exception:
            target_type = str(target_device).strip().lower().split(":", 1)[0]
            if not target_type:
                logger.warning(
                    "[_is_on_target_device] Malformed target device value: {!r}",
                    target_device,
                )
                return False
        return tensor.device.type == target_type

    @staticmethod
    def _get_affine_quantized_tensor_class():
        """Return the AffineQuantizedTensor class from torchao, or None if unavailable."""
        try:
            from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor

            return AffineQuantizedTensor
        except Exception as exc:
            logger.debug(
                "[_get_affine_quantized_tensor_class] failed to import AffineQuantizedTensor from torchao.dtypes.affine_quantized_tensor: {}",
                exc,
            )
        try:
            from torchao.quantization.affine_quantized import AffineQuantizedTensor

            return AffineQuantizedTensor
        except Exception as exc:
            logger.debug(
                "[_get_affine_quantized_tensor_class] failed to import AffineQuantizedTensor from torchao.quantization.affine_quantized: {}",
                exc,
            )
        return None

    def _is_quantized_tensor(self, t):
        """True if ``t`` is a torchao AffineQuantizedTensor."""
        if t is None:
            return False
        cls = self._get_affine_quantized_tensor_class()
        if cls is None:
            return False
        return isinstance(t, cls)

    def _has_quantized_params(self, module):
        """True if module (or any submodule) has an AffineQuantizedTensor parameter."""
        cls = self._get_affine_quantized_tensor_class()
        if cls is None:
            return False
        for _, param in module.named_parameters():
            if param is not None and isinstance(param, cls):
                return True
        return False

    def _ensure_silence_latent_on_device(self):
        """Ensure ``silence_latent`` is on ``self.device``."""
        if hasattr(self, "silence_latent") and self.silence_latent is not None:
            if not self._is_on_target_device(self.silence_latent, self.device):
                self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

    @staticmethod
    def _get_rss_mb() -> float:
        """Return current process RSS in megabytes.

        Uses ``/proc/self/statm`` on Linux for the true current resident set size.
        Uses ``ctypes`` on Windows to call GetProcessMemoryInfo.
        Falls back to ``getrusage`` (peak RSS) on other platforms.
        """
        if platform.system() == "Linux":
            try:
                with open("/proc/self/statm") as f:
                    # statm field index 1 is RSS in pages
                    rss_pages = int(f.read().split()[1])
                # Try os.sysconf for page size, fallback to resource or 4096
                try:
                    page_size = os.sysconf("SC_PAGE_SIZE")
                except (AttributeError, ValueError):
                    page_size = resource.getpagesize() if resource else 4096
                return rss_pages * page_size / (1024 * 1024)
            except Exception as e:
                logger.debug(f"Failed to read RSS from /proc/self/statm; falling back to other methods: {e}")

        if platform.system() == "Windows":
            try:

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
                GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters)):
                    return counters.WorkingSetSize / (1024 * 1024)
            except Exception as exc:
                logger.debug("[memory] Windows RSS query failed: {}", exc)
            return 0.0

        if resource:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if platform.system() == "Darwin":
                # On macOS, ru_maxrss is in bytes
                return usage.ru_maxrss / (1024 * 1024)
            # On other Unixes, ru_maxrss is in kilobytes
            return usage.ru_maxrss / 1024

        return 0.0

    def _release_system_memory(self):
        """Aggressively reclaim system memory after device transfers.

        Combines Python GC, accelerator cache flush, and OS-level heap
        trimming to return freed pages to the operating system.  This is
        critical for CPU-offload workflows where PyTorch ``.to()`` creates
        new tensor storage on each transfer and the old storage may not be
        returned to the OS by the default C allocator.
        """
        gc.collect()
        self._empty_cache()
        libc = _get_libc()
        if libc is not None:
            try:
                libc.malloc_trim(0)
            except Exception as exc:
                logger.debug("[memory] malloc_trim not available or failed: {}", exc)

    def _move_module_recursive(self, module, target_device, dtype=None, visited=None):
        """Recursively move a module and all submodules to the target device."""
        if visited is None:
            visited = set()

        module_id = id(module)
        if module_id in visited:
            return
        visited.add(module_id)

        for param_name, param in module._parameters.items():
            if param is None:
                continue
            if self._is_on_target_device(param, target_device):
                if dtype is not None and param.is_floating_point() and param.dtype != dtype:
                    param.data = param.data.to(dtype)
                continue
            if self._is_quantized_tensor(param):
                module._parameters[param_name] = self._move_quantized_param(param, target_device)
            else:
                new_data = param.data.to(target_device)
                if dtype is not None and new_data.is_floating_point():
                    new_data = new_data.to(dtype)
                param.data = new_data

        for buf_name, buf in module._buffers.items():
            if buf is not None and not self._is_on_target_device(buf, target_device):
                module._buffers[buf_name] = buf.to(target_device)

        for _, child in module._modules.items():
            if child is not None:
                self._move_module_recursive(child, target_device, dtype, visited)

        for attr_name, attr in vars(module).items():
            if attr_name.startswith("_"):
                continue
            if isinstance(attr, torch.nn.Module) and id(attr) not in visited:
                self._move_module_recursive(attr, target_device, dtype, visited)

    def _move_quantized_param(self, param, target_device):
        """Move an AffineQuantizedTensor to target device."""
        if hasattr(param, "_apply_fn_to_data"):
            return torch.nn.Parameter(
                param._apply_fn_to_data(lambda x: x.to(target_device)),
                requires_grad=param.requires_grad,
            )
        moved = param.to(target_device)
        return torch.nn.Parameter(moved, requires_grad=param.requires_grad)

    def _recursive_to_device(self, model, device, dtype=None):
        """Recursively move parameters and buffers to the specified device."""
        target_device = torch.device(device) if isinstance(device, str) else device

        try:
            if dtype is not None:
                model.to(device=target_device, dtype=dtype)
            else:
                model.to(target_device)
        except NotImplementedError:
            logger.info("[_recursive_to_device] model.to() raised NotImplementedError; moving parameters individually.")

        try:
            self._move_module_recursive(model, target_device, dtype)
        except NotImplementedError as exc:
            logger.debug(
                "[_recursive_to_device] _move_module_recursive is not fully supported for this model; "
                "continuing with fallback device placement checks: {}",
                exc,
            )

        if device != "cpu":
            wrong_device_params = []
            for name, param in model.named_parameters():
                if not self._is_on_target_device(param, device):
                    wrong_device_params.append(name)

            if wrong_device_params:
                logger.warning(
                    f"[_recursive_to_device] {len(wrong_device_params)} parameters on wrong device "
                    f"after model.to() + recursive sweep, retrying individually"
                )
                for module in model.modules():
                    for param_name, param in module._parameters.items():
                        if param is None or self._is_on_target_device(param, target_device):
                            continue
                        if self._is_quantized_tensor(param):
                            module._parameters[param_name] = self._move_quantized_param(param, target_device)
                        else:
                            new_data = param.data.to(target_device)
                            if dtype is not None and new_data.is_floating_point():
                                new_data = new_data.to(dtype)
                            param.data = new_data

        if device != "cpu":
            self._synchronize()
