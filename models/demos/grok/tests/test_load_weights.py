from models.demos.grok.tt.model_config import TtModelArgs


def test_load_weights(mesh_device):
    model_args = TtModelArgs(mesh_device)

    state_dict = model_args.load_weights_to_state_dict_no_experts()
    print(state_dict.keys())
    state_dict = model_args.load_experts_weights_to_state_dict(state_dict)
    breakpoint()
