# Stability of id(tensor.device()) (the descriptor memo key) and the stable identity alternatives.
import gc, torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(torch.zeros(32, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    held = [x.device() for _ in range(5)]
    print(
        f"PROBE type={type(device).__name__}/{type(held[0]).__name__} distinct ids while held={len({id(d) for d in held})} same_as_handle={id(device) in {id(d) for d in held}}"
    )
    print(f"PROBE has id()={hasattr(device, 'id')} id={device.id() if hasattr(device, 'id') else None}")
    del held
    gc.collect()
finally:
    ttnn.close_device(device)
