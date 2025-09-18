class Detr3dArgs:
    # Learning rate & scheduler
    base_lr = 5e-4
    warm_lr = 1e-6
    warm_lr_epochs = 9
    final_lr = 1e-6
    lr_scheduler = "cosine"
    weight_decay = 0.1
    filter_biases_wd = False
    clip_gradient = 0.1

    # Model details
    model_name = "3detr"
    enc_type = "masked"  # detr3d-m variant
    enc_nlayers = 3
    enc_dim = 256
    enc_ffn_dim = 128
    enc_dropout = 0.1
    enc_nhead = 4
    enc_pos_embed = None
    enc_activation = "relu"

    dec_nlayers = 8
    dec_dim = 256
    dec_ffn_dim = 256
    dec_dropout = 0.1
    dec_nhead = 4

    mlp_dropout = 0.3
    nsemcls = -1
    preenc_npoints = 2048
    nqueries = 128  # detr3d-m sunrgbd variant
    use_color = False

    # Loss weights
    loss_giou_weight = 0
    loss_sem_cls_weight = 1
    loss_no_object_weight = 0.2
    loss_angle_cls_weight = 0.1
    loss_angle_reg_weight = 0.5
    loss_center_weight = 5.0
    loss_size_weight = 1.0

    # Dataset config
    dataset_name = "sunrgbd"
    dataset_root_dir = "/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/detr3d/sunrgbd_sample_set1"
    meta_data_dir = None
    dataset_num_workers = 4
    batchsize_per_gpu = 8

    # Test / inference
    test_only = True
    test_ckpt = "/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/detr3d/sunrgbd_masked_ep720.pth"
