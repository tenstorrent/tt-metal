# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The experimental device query API (``ttnn.experimental.info``): the uniform, per-coordinate and per-device forms,
the tag objects, and the errors."""

import pytest

import ttnn

info = ttnn.experimental.info


def test_all_tags_lists_every_property():
    assert [tag.name for tag in info.all_tags()] == [
        "l1_alignment",
        "dram_alignment",
        "architecture",
        "architecture_name",
    ]


def test_tags_are_module_attributes_and_compare_by_name():
    for tag in info.all_tags():
        assert isinstance(getattr(info, tag.name), info.InfoTag)
        assert tag == getattr(info, tag.name)
        assert hash(tag) == hash(getattr(info, tag.name))
    assert info.l1_alignment != info.dram_alignment
    assert info.l1_alignment != 5
    assert repr(info.l1_alignment) == "InfoTag(l1_alignment)"


def test_unknown_tag_is_an_attribute_error():
    assert not hasattr(info, "no_such_property")


def test_uniform_values_match_the_hal(mesh_device):
    l1_alignment = info.get_info(mesh_device, info.l1_alignment)
    assert type(l1_alignment) is int
    assert l1_alignment == ttnn._ttnn.device.get_l1_alignment()

    dram_alignment = info.get_info(mesh_device, info.dram_alignment)
    assert type(dram_alignment) is int
    assert dram_alignment == ttnn._ttnn.device.get_dram_alignment()

    arch = info.get_info(mesh_device, info.architecture)
    assert isinstance(arch, ttnn.Arch)

    name = info.get_info(mesh_device, info.architecture_name)
    assert isinstance(name, str)
    assert name == ttnn.get_arch_name()


def test_coordinate_form_matches_uniform_form(mesh_device):
    expected = info.get_info(mesh_device, info.l1_alignment)
    rows, cols = list(mesh_device.shape)
    last = ttnn.MeshCoordinate(rows - 1, cols - 1)
    assert info.get_info(mesh_device, info.l1_alignment, ttnn.MeshCoordinate(0, 0)) == expected
    assert info.get_info(mesh_device, info.l1_alignment, coord=last) == expected


def test_per_device_has_one_entry_per_local_device(mesh_device):
    expected = info.get_info(mesh_device, info.l1_alignment)
    per_device = info.get_info_per_device(mesh_device, info.l1_alignment)
    assert isinstance(per_device, dict)
    assert len(per_device) == mesh_device.get_num_devices()
    assert all(isinstance(coord, ttnn.MeshCoordinate) for coord in per_device)
    assert all(value == expected for value in per_device.values())

    per_device_arch = info.get_info_per_device(mesh_device, info.architecture)
    assert all(isinstance(value, ttnn.Arch) for value in per_device_arch.values())


@pytest.mark.parametrize("tag_name", ["l1_alignment", "architecture_name"])
def test_out_of_bounds_coordinate_names_the_property(mesh_device, tag_name):
    outside = ttnn.MeshCoordinate(list(mesh_device.shape)[0], 0)
    with pytest.raises(RuntimeError, match=f"Cannot query {tag_name} at"):
        info.get_info(mesh_device, getattr(info, tag_name), outside)


def test_a_string_is_not_a_tag(mesh_device):
    with pytest.raises(TypeError):
        info.get_info(mesh_device, "l1_alignment")
