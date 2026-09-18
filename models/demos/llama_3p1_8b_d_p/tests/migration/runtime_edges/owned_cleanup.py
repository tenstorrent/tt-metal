"""Attempt each owned release independently; preserve errors for root recovery review."""


def cleanup_each(actions, errors, attempts):
    for label, action in actions:
        attempts.append(label)
        try:
            action()
        except BaseException as exc:
            errors.append(label + ": " + repr(exc))


def cleanup_owner(owned, *, api, collect, errors):
    attempts = []

    def run(label, action):
        cleanup_each(((label, action),), errors, attempts)

    # Drop the persistent H2D service before cache ownership. Its destructor handles
    # pending consumer waits; an unconditional full service barrier can hang.
    run("service.drop", lambda: owned.pop("service", None))
    run("service.collect", collect)
    run("channel.drop", lambda: owned.pop("channel", None))
    for key, method in (
        ("producer", lambda obj: obj.shutdown()),
        ("router", lambda obj: obj.stop()),
        ("saved", lambda obj: obj.close()),
        ("cache.k", lambda obj: obj.deallocate(True)),
        ("cache.v", lambda obj: obj.deallocate(True)),
        ("model", lambda obj: obj.close()),
    ):
        if owned.get(key) is not None:
            run(key, lambda key=key, method=method: method(owned.pop(key)))
    run("model.collect", collect)
    mesh = owned.get("mesh")
    if mesh is not None:
        run("mesh.synchronize", lambda: api.synchronize_device(mesh))
        run("fabric.disable", lambda: api.set_fabric_config(api.FabricConfig.DISABLED))
        # This is the last explicit owned action even if any earlier action failed.
        run("mesh.close", lambda: api.close_mesh_device(mesh))
    return attempts
