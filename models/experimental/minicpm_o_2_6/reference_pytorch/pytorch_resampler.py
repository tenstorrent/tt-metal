try:
    # Preferred: official implementation in minicpm_official (absolute import)
    from minicpm_official.resampler import get_2d_sincos_pos_embed, Resampler as PyTorchResampler
except Exception:
    # Fallback to legacy reference_incorrect if present on sys.path
    try:
        # Import directly from the reference directory to avoid circular import
        import sys
        import os

        reference_dir = os.path.join(os.path.dirname(__file__), "..", "reference")
        if reference_dir not in sys.path:
            sys.path.insert(0, reference_dir)
        from pytorch_resampler import get_2d_sincos_pos_embed, PyTorchResampler
    except Exception:
        raise

__all__ = ["get_2d_sincos_pos_embed", "PyTorchResampler"]
