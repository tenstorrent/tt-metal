"""Fail-closed compile-consumer selection for an explicitly staged variant."""

from __future__ import annotations

from pathlib import Path


def resolve_staged_variant(
    generated_variant: str,
    explicit_variant: str | None,
    *,
    compile_consumer: bool,
) -> str:
    """Use an identity-gated staged id without trusting a host-local hash.

    Variant generation includes environment/path-dependent configuration. A
    producer and a consumer can therefore name identical source/configuration
    differently. The exhaustive streamer supplies the producer id from its
    arm-specific identity map. This override is legal only in consumer mode.
    """
    if explicit_variant is None:
        return generated_variant
    if not compile_consumer:
        raise RuntimeError(
            "TT_LLK_STAGED_VARIANT_ID is valid only with --compile-consumer"
        )
    if len(explicit_variant) != 64 or any(
        char not in "0123456789abcdef" for char in explicit_variant
    ):
        raise RuntimeError("TT_LLK_STAGED_VARIANT_ID must be 64 lowercase hex digits")
    return explicit_variant


def require_complete_staged_variant(
    elf_dir: Path, components: list[str], variant: str
) -> None:
    missing = [name for name in components if not (elf_dir / f"{name}.elf").is_file()]
    if missing:
        raise RuntimeError(
            "explicit staged variant is incomplete: "
            f"variant={variant}, missing={missing}"
        )
