from models.experimental.opt_transfer.assemble import assemble_model


def test_assemble_chains_blocks_in_order():
    calls = []

    def make_block(i):
        def run(x):
            calls.append(i)
            return x + i

        return run

    model = assemble_model([make_block(1), make_block(2), make_block(3)])
    assert model(0) == 6
    assert calls == [1, 2, 3]


def test_assemble_empty_is_identity():
    model = assemble_model([])
    assert model(42) == 42


from models.experimental.opt_transfer.assemble import assemble_layers


def test_assemble_layers_stacks_in_order():
    m = assemble_layers([lambda x: x + 1, lambda x: x * 2, lambda x: x - 3])
    assert m(3) == ((3 + 1) * 2) - 3
