import ttnn

device = ttnn.open_device(device_id=0)


def v(x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return (int(c.x), int(c.y))


# 8x4 logical rectangle virtual coords
for gx, gy, label in [(8, 1, "8x1"), (9, 1, "9x1"), (8, 4, "8x4"), (7, 4, "7x4"), (8, 8, "8x8 row0")]:
    members = [v(x, y) for y in range(gy) for x in range(gx)]
    xs = [a for a, b in members]
    ys = [b for a, b in members]
    area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    print(
        f"{label}: nmembers={len(members)} vbox x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}] area={area} rect_ok={area==len(members)}"
    )
    print("   sample vcoords:", members[:10])
ttnn.close_device(device)
