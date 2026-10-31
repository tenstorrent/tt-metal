"""Host-only exact-context admission with inactive fixed slots."""

from types import SimpleNamespace

import torch

from ..tt.generator import Gemma4Generator


def main():
    gen = object.__new__(Gemma4Generator)
    gen.owns_cache = True
    gen.model = SimpleNamespace(pool_blocks=8192, layers=[SimpleNamespace(sliding_window=None)])
    gen.prefill = SimpleNamespace(host_pages=[torch.zeros(8, 8192, dtype=torch.int32)])
    observed = {}

    def refresh(maps, *, revision):
        observed.update(maps=maps, revision=revision)

    gen.refresh_page_tables = refresh
    gen._reserve_requests([0, 0, 262143, 0, 0, 0, 0, 0], 1)
    assert observed["revision"] == (0, 0, 8192, 0, 0, 0, 0, 0)
    assert torch.equal(observed["maps"][0][2], torch.arange(8192, dtype=torch.int32))
    assert all(bool((observed["maps"][0][i] == 0).all()) for i in (0, 1, 3, 4, 5, 6, 7))
    try:
        gen._reserve_requests([128, 0, 262143, 0, 0, 0, 0, 0], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Active overcommit must be rejected")
    gen.batch, gen.context = 8, 262144
    gen.model.config = SimpleNamespace(vocab_size=262144)
    gen.kv_cache, gen.page_table = object(), object()
    for tokens, positions in (
        (torch.zeros(8, dtype=torch.int32), torch.full((8,), -2)),
        (torch.zeros(8, dtype=torch.int32), torch.full((8,), 262144)),
        (torch.zeros(7, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)),
        (torch.full((8,), 262144), torch.zeros(8, dtype=torch.int32)),
        (torch.zeros(8, dtype=torch.int32), torch.zeros(8)),
    ):
        try:
            gen.decode_forward(tokens, positions, page_table=gen.page_table, kv_cache=gen.kv_cache)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid decode state must fail before device work")
    gen.sampler = SimpleNamespace(reset_params=lambda k, p, t: observed.update(k=k, p=p, t=t), seeds_tt_tensor=None)
    gen._copy = lambda *args: None
    gen.set_sampling(temperature=0.8, top_k=32, top_p=0.9, seed=101)
    assert observed["k"] == [32] * 8 + [1] * 24
    assert observed["p"] == [0.9] * 8 + [0.0] * 24
    assert observed["t"] == [1.25] * 8 + [1.0] * 24
    for bad in (dict(temperature=0.8, top_k=64), dict(temperature=0.8, top_k=0), dict(temperature=[0.8])):
        try:
            gen.set_sampling(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported sampling parameters must be rejected")
    print("Exact262144-token request fits an8192-page pool with7 inactive slots; active overcommit rejected")


if __name__ == "__main__":
    main()
