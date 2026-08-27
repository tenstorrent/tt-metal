# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sentences per language, as data, tokenized at test time by the in-repo tokenizer.

`prompt_fixture.json` stores text WITH the token ids it produced and is the bit-exactness oracle;
this module is plain text, so breadth costs nothing.

Conventions: no digits in either form (the cleaner expands them and a recogniser writes them back as
numerals), ordinary prose, Arabic undiacritized, and sentences long enough to gate on.
"""

# Keyed by the language prefix of a voice name; "en" covers the unprefixed voices
# (neutral_*, casual_*, cheerful_*).
SENTENCES = {
    "en": [
        "It took me quite a long time to develop a voice, and now that I have it I am not going to be silent.",
        "The morning light came through the tall windows and settled on the wooden floor without a sound.",
        "She explained the whole arrangement twice, patiently, until everyone at the table understood it.",
    ],
    "fr": [
        "Il m'a fallu longtemps pour trouver ma voix, et maintenant que je l'ai, je ne me tairai pas.",
        "La lumière du matin traversait les hautes fenêtres et se posait doucement sur le plancher.",
    ],
    "de": [
        "Ich habe lange gebraucht, um meine Stimme zu finden, und jetzt werde ich nicht mehr schweigen.",
        "Das Morgenlicht fiel durch die hohen Fenster und legte sich lautlos auf den hölzernen Boden.",
    ],
    "es": [
        "Me llevó mucho tiempo encontrar mi voz, y ahora que la tengo no voy a quedarme callada.",
        "La luz de la mañana entraba por las ventanas altas y se posaba en el suelo de madera.",
    ],
    "it": [
        "Mi è servito molto tempo per trovare la mia voce, e ora che l'ho trovata non resterò in silenzio.",
        "La luce del mattino entrava dalle finestre alte e si posava senza rumore sul pavimento di legno.",
    ],
    "pt": [
        "Levei muito tempo para encontrar a minha voz, e agora que a tenho não vou ficar calada.",
        "A luz da manhã entrava pelas janelas altas e assentava sem ruído no chão de madeira.",
    ],
    "nl": [
        "Het heeft me lang gekost om mijn stem te vinden, en nu ik hem heb zal ik niet zwijgen.",
        "Het ochtendlicht viel door de hoge ramen en legde zich geluidloos op de houten vloer.",
    ],
    "hi": [
        "मुझे अपनी आवाज़ पाने में बहुत समय लगा, और अब जब मेरे पास है तो मैं चुप नहीं रहूँगी।",
        "सुबह की रोशनी ऊँची खिड़कियों से आकर लकड़ी के फ़र्श पर बिना आवाज़ ठहर गई।",
    ],
    "ar": [
        "استغرق الأمر وقتا طويلا حتى وجدت صوتي، والآن بعد أن وجدته لن أصمت.",
        "كان ضوء الصباح يعبر النوافذ العالية ويستقر على الأرضية الخشبية بلا صوت.",
    ],
}


def lang_of(voice):
    """Voice name -> corpus key. `ar_male` -> "ar"; `neutral_male` -> "en"."""
    head = voice.split("_")[0]
    return head if head in SENTENCES else "en"


def sentences_for(voice):
    return SENTENCES[lang_of(voice)]


def first_sentence_for(voice):
    return sentences_for(voice)[0]
