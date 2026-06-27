"""Strategist (axis selection) — choose_axis + prompt (no hardware)."""

from agent.strategist import build_axis_prompt, choose_axis

_PROFILE = {
    "wall_ms": 27000.0,
    "device_ms": 90.0,
    "buckets": [
        {"id": "datamove", "device_ms": 57.0},
        {"id": "matmul", "device_ms": 27.0},
        {"id": "host_overhead", "device_ms": 26910.0},
    ],
}


def test_prompt_shows_wall_device_host_breakdown():
    p = build_axis_prompt(_PROFILE)
    assert "wall_ms" in p and "device_ms" in p and "host_overhead" in p
    assert "datamove" in p and "matmul" in p  # device buckets surfaced
    # host_overhead is NOT listed as a device bucket (it's a host artifact, not a device op-class)
    bucket_list = p.split("top device buckets:")[-1]
    assert "datamove:" in bucket_list and "host_overhead:" not in bucket_list


def test_choose_axis_host_maps_to_wall_ms():
    # host overhead dominates -> agent picks 'host' -> optimize wall_ms (activates host levers)
    out = choose_axis(_PROFILE, lambda prompt: {"axis": "host", "reasoning": "host is 99.7%"})
    assert out == "wall_ms"


def test_choose_axis_device_maps_to_device_ms():
    out = choose_axis(_PROFILE, lambda prompt: {"axis": "device", "reasoning": "kernels have headroom"})
    assert out == "device_ms"


def test_choose_axis_accepts_json_text():
    out = choose_axis(_PROFILE, lambda prompt: '{"axis": "host"}')
    assert out == "wall_ms"


def test_choose_axis_falls_back_on_junk():
    assert choose_axis(_PROFILE, lambda prompt: "not json at all") == "device_ms"
    assert choose_axis(_PROFILE, lambda prompt: {"axis": "sideways"}) == "device_ms"


def test_choose_axis_falls_back_on_runner_error():
    def boom(prompt):
        raise RuntimeError("api down")

    assert choose_axis(_PROFILE, boom) == "device_ms"
