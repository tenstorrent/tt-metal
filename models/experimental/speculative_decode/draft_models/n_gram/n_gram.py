#!/bin/env python

import argparse
import math
import nltk
from pathlib import Path

import time
import itertools
import numpy as np


def load_data(data_dir):
    """Load train and test corpora from a directory.

    Directory must contain two files: train.txt and test.txt.
    Newlines will be stripped out.

    Args:
        data_dir (Path) -- pathlib.Path of the directory to use.

    Returns:
        The train and test sets, as lists of sentences.

    """
    train_path = data_dir.joinpath("train.txt").absolute().as_posix()
    test_path = data_dir.joinpath("test.txt").absolute().as_posix()

    with open(train_path, "r") as f:
        train = [l.strip() for l in f.readlines()]
    with open(test_path, "r") as f:
        test = [l.strip() for l in f.readlines()]
    return train, test


def encode_sentence(data, tockenizer, n):
    tokens = tockenizer.encode(data, bos=True, eos=True)
    if n > 2:
        tokens = [tockenizer.bos_id for _ in range(n - 2)] + tokens
    return tokens


from collections import defaultdict


class NGramModel(object):
    def __init__(self, train_data, n, tokenizer, laplace=1):
        start_time = time.time()
        self.n = n
        self.tokenizer = tokenizer
        self.laplace = laplace
        self.tokens = list(
            itertools.chain.from_iterable([encode_sentence(train_data_i, tokenizer, n) for train_data_i in train_data])
        )
        self.vocab, self.vocab_counts = np.unique(self.tokens, return_counts=True)
        self.model = self._create_model()
        print("N-Gram model created in {:.2f} seconds".format(time.time() - start_time))

    def _smooth(self, n_vocab, m_vocab, vocab_size):
        smoothed_probs = defaultdict(dict)
        for n_gram, n_count in n_vocab.items():
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            prob = (n_count + self.laplace) / (m_count + self.laplace * vocab_size)
            smoothed_probs[m_gram][n_gram[-1]] = prob
        return smoothed_probs

    def _create_model(self):
        if self.n == 1:
            num_tokens = len(self.tokens)
            return {(unigram,): count / num_tokens for unigram, count in zip(self.vocab, self.vocab_counts)}
        else:
            n_grams = nltk.ngrams(self.tokens, self.n)
            m_grams = nltk.ngrams(self.tokens, self.n - 1)

            n_vocab = nltk.FreqDist(n_grams)
            m_vocab = nltk.FreqDist(m_grams)
            vocab_size = len(self.vocab)

            return self._smooth(n_vocab, m_vocab, vocab_size)

    def _best_candidate(self, prev, without=[], sampling=True):
        """Choose the most likely next token given the previous (n-1) tokens.

        Always selects the candidate with the highest probability.
        If no candidates are found, the EOS token is returned with probability 1.

        Args:
            prev (tuple of str): the previous n-1 tokens of the sentence.
            without (list of str): tokens to exclude from the candidates list.

        Returns:
            A tuple with the next most probable token and its corresponding probability.
        """
        if prev in self.model:
            candidates = {token: prob for token, prob in self.model[prev].items() if token not in without}
            if candidates:
                if sampling:
                    tokens = list(candidates.keys())
                    probabilities = np.array(list(candidates.values()))
                    probabilities /= probabilities.sum()  # Normalize to form a valid probability distribution
                    chosen_token = np.random.choice(tokens, p=probabilities)
                    return (chosen_token, candidates[chosen_token])
                else:
                    best_candidate = max(candidates.items(), key=lambda x: x[1])
                    return best_candidate

        return (self.tokenizer.eos_id, 0)

    def __call__(self, sent, generation_length=5, sampling=True):
        """Continue generate n tokens based on history.

        Args:
            sent (list): token ids of the sentence.
        Returns:
            newly generated tokens in a list

        """
        sent = sent.copy()
        probabilities = []

        if len(sent) < self.n - 1:
            sent = [self.tokenizer.bos_id] * (self.n - 1 - len(sent)) + sent

        for _ in range(generation_length):
            prev = () if self.n == 1 else tuple(sent[-(self.n - 1) :])
            blacklist = [self.tokenizer.eos_id] if len(sent) < generation_length else []
            next_token, p = self._best_candidate(prev, without=blacklist, sampling=sampling)
            sent.append(next_token)
            probabilities.append(p)

        return sent[-generation_length:], probabilities

    def generate_sentences_demo(self, num, min_len=12, max_len=24, sampling=True):
        """Generate num random sentences using the language model.

        Sentences always begin with the SOS token and end with the EOS token.

        Args:
            num (int): the number of sentences to generate.
            min_len (int): minimum allowed sentence length.
            max_len (int): maximum allowed sentence length.
        Yields:
            A tuple with the generated sentence and the combined probability
            (in log-space) of all of its n-grams.

        """
        total_time = 0
        total_count = 0
        for i in range(num):
            # record how long it takes to generate a token
            sent, _ = [self.tokenizer.bos_id] * max(1, self.n - 1), 1
            while sent[-1] != self.tokenizer.eos_id:
                time_start = time.time()
                prev = () if self.n == 1 else tuple(sent[-(self.n - 1) :])
                blacklist = [self.tokenizer.eos_id] if len(sent) < min_len else []
                next_token, _ = self._best_candidate(prev, without=blacklist, sampling=sampling)
                sent.append(next_token)

                if len(sent) >= max_len:
                    sent.append(self.tokenizer.eos_id)
                time_end = time.time()
                total_time += time_end - time_start
                total_count += 1

            yield self.tokenizer.decode(sent), 0
        print("Average time to generate a token: ", total_time / total_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("N-gram Language Model")
    parser.add_argument(
        "--n", type=int, required=True, help="Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)"
    )
    parser.add_argument(
        "--laplace",
        type=float,
        default=0.01,
        help="Lambda parameter for Laplace smoothing (default is 0.01 -- use 1 for add-1 smoothing)",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["sample-data", "wikitext-103"],
        default="sample-data",
        help="Directory containing train/test data",
    )
    parser.add_argument("--num", type=int, default=10, help="Number of sentences to generate (default 10)")
    args = parser.parse_args()

    # Load and prepare train/test data
    train, test = None, None
    if args.data == "sample-data":
        data_path = Path("models/experimental/speculative_decode/draft_models/n_gram/sample_data")
        train, test = load_data(data_path)

    elif args.data == "wikitext-103":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", ignore_verifications=True)
        train, test = ds["train"]["text"], ds["test"]["text"]
        # remove empty sentences ''
        train = [s for s in train if s != ""]
    else:
        raise ValueError("Invalid data choice")

    # Load the Tokenizer
    from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
    from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs

    model_args = TtModelArgs(None, instruct=True)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    print("Loading {}-gram model...".format(args.n))
    lm = NGramModel(train, args.n, tokenizer, laplace=args.laplace)
    print("Vocabulary size: {}".format(len(lm.vocab)))

    print("Generating sentences...")
    for sentence, prob in lm.generate_sentences_demo(args.num):
        print("{} ({:.5f})".format(sentence, prob))
