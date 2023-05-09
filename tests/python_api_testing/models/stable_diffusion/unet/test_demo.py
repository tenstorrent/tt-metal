from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


from torch import autocast
import torch
import time
from PIL import Image


from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm

from utility_functions import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc
from libs import tt_lib as ttl
from python_api_testing.models.stable_diffusion.unet.unet_2d_condition import UNet2DConditionModel as tt_unet_condition

def constant_prop_time_embeddings(timesteps, sample, time_proj):
    # self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb




def demo():

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # 1. Load t`he autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    torch_unet = unet
    # 4. load the K-LMS scheduler with some fitting parameters.
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    #scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    #scheduler = HeunDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    #scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    torch_device = "cpu"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)
    print(unet)
    state_dict = torch_unet.state_dict()
    tt_unet = tt_unet_condition(sample_size = 64,
                                in_channels = 4,
                                out_channels = 4,
                                center_input_sample = False,
                                flip_sin_to_cos = True,
                                freq_shift = 0,
                                down_block_types = ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
                                mid_block_type = 'UNetMidBlock2DCrossAttn',
                                up_block_types = ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
                                only_cross_attention = False,
                                block_out_channels = [320, 640, 1280, 1280],
                                layers_per_block = 2,
                                downsample_padding = 1,
                                mid_block_scale_factor = 1,
                                act_fn = 'silu',
                                norm_num_groups = 32,
                                norm_eps = 1e-05,
                                cross_attention_dim = 768,
                                attention_head_dim = 8,
                                dual_cross_attention = False,
                                use_linear_projection = False,
                                class_embed_type = None,
                                num_class_embeds = None,
                                upcast_attention = False,
                                resnet_time_scale_shift = 'default',
                                state_dict=state_dict,
                                base_address="")

    tt_unet.config = torch_unet.config
    prompt = ["a photo of an astronaut riding a horse on mars"]
    prompt = ["oil painting frame of Breathtaking mountain range with a clear river running through it, surrounded by tall trees and misty clouds, serene, peaceful, mountain landscape, high detail"]

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 1           # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(174)    # 10233 Seed generator to create the inital latent noise
    batch_size = len(prompt)

    tic= time.time()
    ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
    # Tokenizer and Text Encoder
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    #For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
    # and another with the unconditional embeddings (uncond_embeddings).
    # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    toc= time.time()
    print("Tokenizer Time:", round(toc-tic, 3))

    # Denoising loop
    scheduler.set_timesteps(num_inference_steps)
    tic= time.time()

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        t1 = time.time()
        # predict the noise residual
        with torch.no_grad():
            _t = constant_prop_time_embeddings(t, latent_model_input, torch_unet.time_proj)

            _t = torch_to_tt_tensor_rm(_t, device, put_on_device=False)
            tt_latent_model_input = torch_to_tt_tensor_rm(latent_model_input, device, put_on_device=False)
            tt_text_embeddings = torch_to_tt_tensor_rm(text_embeddings, device, put_on_device=False)

            tt_noise_pred = tt_unet(tt_latent_model_input, _t, encoder_hidden_states=tt_text_embeddings)

            torch_noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

        tt_noise_pred = tt_to_torch_tensor(tt_noise_pred, host)
        print("########### comparing ############")
        print(comp_allclose_and_pcc(torch_noise_pred, tt_noise_pred))
        print("######### end of comparing #########")

        noise_pred = tt_noise_pred


        t2 = time.time()
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        t3= time.time()

        print( "/t", round(t2-t1, 2), " | ", round(t3-t2, 2), " | ", round(t3-t1, 2) )
    toc = time.time()
    print("UNET Loop Time:", round(toc-tic, 3))


    tic = time.time()
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    pil_images.save("mountain_river_oil_painintg__512.png")
    toc = time.time()
    print("Image Time:", round(toc-tic, 3))

'''
@article{patil2022stable,
author = {Patil, Suraj and Cuenca, Pedro and Lambert, Nathan and von Platen, Patrick},
title = {Stable Diffusion with :firecracker: Diffusers},
journal = {Hugging Face Blog},
year = {2022},
note = {[https://huggingface.co/blog/rlhf](https://huggingface.co/blog/stable_diffusion)},
}
'''
demo()
