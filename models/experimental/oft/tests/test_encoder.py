import torch
from models.experimental.oft.reference.encoder import ObjectEncoder


def make_grid(grid_size, grid_offset, grid_res):
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset
    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff
    zz, xx = torch.meshgrid(zcoords, xcoords, indexing="ij")
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


def test_decode():
    encoder = ObjectEncoder(classnames=["Car"])
    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    # Prepare dummy inputs
    scores = torch.rand((1, 1, 159, 159), dtype=torch.float32)
    scores[0, 0, 80, 80] = 0.9  # single peak
    pos_offsets = torch.rand((1, 1, 3, 159, 159), dtype=torch.float32)
    dim_offsets = torch.rand((1, 1, 3, 159, 159), dtype=torch.float32)
    ang_offsets = torch.rand((1, 1, 2, 159, 159), dtype=torch.float32)
    objects = encoder.decode(scores[0], pos_offsets[0], dim_offsets[0], ang_offsets[0], grid)
    obj = objects[0]
    assert obj.classname == "Car"
    assert torch.is_tensor(obj.position)
    assert torch.is_tensor(obj.dimensions)
    assert isinstance(obj.angle, torch.Tensor)
    assert obj.score > 0.8


def test_decode_batch():
    encoder = ObjectEncoder(classnames=["Car", "Pedestrian"])
    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    scores = torch.zeros((2, 2, 159, 159), dtype=torch.float32)
    scores[0, 0, 50, 50] = 0.7
    scores[1, 1, 100, 100] = 0.8
    pos_offsets = torch.zeros((2, 2, 3, 159, 159), dtype=torch.float32)
    dim_offsets = torch.zeros((2, 2, 3, 159, 159), dtype=torch.float32)
    ang_offsets = torch.zeros((2, 2, 2, 159, 159), dtype=torch.float32)
    grids = [grid, grid]
    boxes = encoder.decode_batch(scores, pos_offsets, dim_offsets, ang_offsets, grids)
    assert len(boxes) == 2
    assert any(obj.classname == "Car" for obj in boxes[0])
    assert any(obj.classname == "Pedestrian" for obj in boxes[1])


def test_nms_multiple_peaks():
    encoder = ObjectEncoder(classnames=["Car"])
    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    scores = torch.zeros((1, 1, 159, 159), dtype=torch.float32)
    scores[0, 0, 10, 10] = 0.6
    scores[0, 0, 20, 20] = 0.7
    scores[0, 0, 30, 30] = 0.8
    pos_offsets = torch.zeros((1, 1, 3, 159, 159), dtype=torch.float32)
    dim_offsets = torch.zeros((1, 1, 3, 159, 159), dtype=torch.float32)
    ang_offsets = torch.zeros((1, 1, 2, 159, 159), dtype=torch.float32)
    objects = encoder.decode(scores[0], pos_offsets[0], dim_offsets[0], ang_offsets[0], grid)
    assert len(objects) >= 1
    assert all(obj.score >= 0.6 for obj in objects)
