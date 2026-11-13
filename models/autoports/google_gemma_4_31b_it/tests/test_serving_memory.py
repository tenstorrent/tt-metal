# SPDX-License-Identifier: Apache-2.0
from models.autoports.google_gemma_4_31b_it.tt.serving_memory import validate_pool_memory


def plan(free_per_bank=3453085824, contiguous_per_bank=3431885696):
    specs = [(i // 6, (36896, 4, 32, 256) if i % 6 != 5 else (36896, 1, 64, 512), 1088) for i in range(60)]
    return validate_pool_memory(
        allocator={
            "num_banks": 8,
            "total_bytes_free_per_bank": free_per_bank,
            "largest_contiguous_bytes_free_per_bank": contiguous_per_bank,
        },
        device_ids=[0, 1, 2, 3],
        cache_specs=specs,
        batch=32,
        context=262144,
    )


def test_shared_hybrid_pool_reconciles_ledger():
    result = plan()
    assert result["fits"]
    assert len(result["per_device"]) == 4
    assert result["per_device"][0]["pool_bytes"] == 25691422720
    assert result["page_map_bytes"] == 62914560
    assert result["per_device"][0]["reserve_bytes"] == 1015693824


def test_insufficient_free_space_rejected():
    assert not plan(free_per_bank=3200000000)["fits"]


def test_fragmentation_rejected_even_when_total_free_suffices():
    assert not plan(contiguous_per_bank=3200000000)["fits"]
