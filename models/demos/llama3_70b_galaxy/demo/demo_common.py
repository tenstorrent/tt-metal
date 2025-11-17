# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utility functions for demo scripts.
"""

import json
import hashlib
import requests
from pathlib import Path
from loguru import logger


def load_and_cache_context(context_url, cache_dir, max_length=None):
    """
    Load and cache context from a URL.

    Args:
        context_url (str): URL to fetch context from
        cache_dir (Path): Directory to cache the context
        max_length (int, optional): Maximum length to clip the context to

    Returns:
        str: The context text (possibly clipped)
    """
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        with open(cache_file, "r") as f:
            context_text = f.read()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            response = requests.get(context_url)
            if response.status_code == 200:
                context_text = response.text
                with open(cache_file, "w") as f:
                    f.write(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from URL: {context_url}. Status code: {response.status_code}")
                context_text = ""
        except Exception as e:
            logger.error(f"Error fetching context from URL: {context_url}. Error: {str(e)}")
            context_text = ""

    # Clip the context to the max length provided
    if max_length:
        context_text = context_text[:max_length]
        logger.info(f"Clipped the context text to {max_length} characters")

    return context_text


def load_inputs_simple(user_input, batch, instruct_mode, cache_dir):
    """
    Load inputs for simple demo (decode demos).

    Args:
        user_input (str or list): Path to JSON file or list of input data
        batch (int): Number of users in batch
        instruct_mode (bool): Whether to use instruct mode (currently unused)
        cache_dir (str or Path): Path to the cache directory

    Returns:
        list: List of input prompts
    """
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(batch):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            if "max_length" in user_input[i]:  # Clip the context to the max length provided
                context_text = load_and_cache_context(
                    user_input[i]["context"], cache_dir, max_length=user_input[i]["max_length"]
                )
            else:
                context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            prompt = context_text
        in_prompt.append(prompt)
    return in_prompt


def load_inputs_advanced(user_input, len_per_batch, instruct, cache_dir):
    """
    Load inputs for advanced demo (text demos with repeat batches).

    Args:
        user_input (str or list): Path to JSON file or list of input data
        len_per_batch (list): List of lengths per batch
        instruct (bool): Whether to use instruct mode
        cache_dir (str or Path): Path to the cache directory

    Returns:
        tuple: (in_prompt, all_prompts) - prompts for current batch and all prompts
    """
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    batch = len(len_per_batch)
    user_input = user_input * batch
    in_prompt = []
    all_prompts = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The demo supports a custom prompt file, where the context is provided by a link to a book from the gutenberg project
    # It clips the excerpt to the max length provided to allow testing different long context lengths
    for i in range(len(user_input)):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            # TODO This might override the expected input size given in the prompt file
            if "max_length" in user_input[i]:  # Clip the context to the max length provided
                context_text = load_and_cache_context(
                    user_input[i]["context"],
                    cache_dir,
                    max_length=(user_input[i]["max_length"]) if batch == 1 else len_per_batch[i],
                )
            else:
                context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            if instruct:
                prompt = (
                    "```" + context_text + "```\n\n" + prompt
                )  # Add the markdown block to the context to comply with the prompt
            else:
                prompt = context_text

        all_prompts.append(prompt)  # return all the prompts taken from the input file to be used when repeat_batch > 1
        if i in range(batch):
            in_prompt.append(prompt)

    return in_prompt, all_prompts
