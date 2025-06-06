# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import open_clip
from PIL import Image
from typing import Union, List, Optional


class CLIPEncoder(nn.Module):
    """
    A class for encoding images and texts using a specified CLIP model and computing the similarity between them.

    Attributes:
    -----------
    clip_version: str
        The version of the CLIP model to be used.
    pretrained: str
        The pre-trained weights to load.
    model: nn.Module
        The CLIP model.
    preprocess: Callable
        The preprocessing transform to apply to the input image.
    device: str
        The device to which the model is moved.
    """

    def __init__(
        self,
        clip_version: str = "ViT-B/32",
        pretrained: Optional[str] = "",
        cache_dir: Optional[str] = None,
    ):
        """
        Initializes the CLIPEncoder with the specified CLIP model version and pre-trained weights.

        Parameters:
        -----------
        clip_version: str, optional
            The version of the CLIP model to be used. Defaults to 'ViT-B/32'.
        pretrained: str, optional
            The pre-trained weights to load. If not provided, it defaults based on clip_version.
        cache_dir: str, optional
            The directory to cache the model. Defaults to None.
        device: str, optional
            The device to which the model is moved. Defaults to 'cuda'.
        """
        super().__init__()

        self.clip_version = clip_version
        self.pretrained = pretrained if pretrained else self._get_default_pretrained()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_version, pretrained=self.pretrained, cache_dir=cache_dir
        )

        self.model.eval()

    def _get_default_pretrained(self) -> str:
        """Returns the default pretrained weights based on the clip_version."""
        return "openai"

    @torch.no_grad()
    def get_clip_score(self, text: Union[str, List[str]], image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Computes the similarity score between the given text(s) and image using the CLIP model.

        Parameters:
        -----------
        text: Union[str, List[str]]
            The text or list of texts to compare with the image.
        image: Image.Image
            The input image.

        Returns:
        --------
        torch.Tensor
            The similarity score between the text(s) and image.
        """
        # Preprocess the image and move it to the specified device
        image = self.preprocess(image).unsqueeze(0)

        # Normalize the image features
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # If a single text string is provided, convert it to a list
        if not isinstance(text, (list, tuple)):
            text = [text]

        # Tokenize the text and move it to the specified device
        text = open_clip.tokenize(text)

        # Normalize the text features
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity between the image and text features
        similarity = image_features @ text_features.T

        return similarity
